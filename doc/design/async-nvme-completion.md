# GPU 直连 NVMe 的异步完成：提交与回收分离

> **状态**：正常路径已实现（2026-10-09，实验开关 `TUTTI_ASYNC_COMPLETION=1`，默认关闭），实测见 §11。
>
> **2026-10-10 生命周期修正**：异步超时不再提前结束 kernel/批次；保持在飞
> 状态直到实际完成和全部等待流退休。关闭超时保留资源且可重试，Runtime 保留全部
> 子操作的内存/目标引用。新增 GPU 显存合成 CQ、迟到完成、多流等待和软件释放失败
> 测试，见 §12。默认仍关闭；修正后的真实 NVMe/LLM 回归尚未重跑。
> 同步 DataPath 原有 timeout 路径不在本轮改动范围。
>
> 本文讨论把 GPU 发起的 NVMe IO 从"IO kernel 提交后原地等自己的完成"改成"提交
> kernel 写完命令就退出，由回收者统一处理完成队列"。重点回答两个问题：
>
> 1. **等待者怎么知道某一层（某个 batch）提交的所有命令都完成了？**
> 2. **cid（NVMe 命令号）和 batch IO、和 Python 拿到的 handle 是怎么关联起来的？**
>
> 文中还给出无死锁论证、超时处理、接入方式（不改动现有函数）和验证计划。

相关文档：[`kv-storage-layout-and-io-submission.md`](kv-storage-layout-and-io-submission.md)
（现有提交 kernel、worker 池、GEMM 占满 SM 的根因）、
[`tutti-runtime-assembly.md`](tutti-runtime-assembly.md)（Runtime / DataPath 装配）。

### 术语与命名

文中新增的名字都按"看名字就知道是什么"来取，不用缩写和行话：

| 名字 | 含义 |
|---|---|
| 批次（batch） | 一次 `submit()` 被接受的那组条目，在 DataPath 里叫一个 op，占用一个 arena 槽位。Python 一层的读通常就是一个批次 |
| 条目（entry） | 批次里的一条 IO，对应一条 NVMe 命令 |
| 槽位（slot） | arena 里的一块固定工作区（entries、status 数组等），批次结束后给下一个批次复用 |
| `batch_id` | 批次号。主机每接受一个批次就分配一个新号，单调递增、不重复。用来区分同一个槽位上先后跑过的不同批次 |
| `pending_entries[slot]` | 该槽位当前批次还没结束的条目数 |
| `finished_batch_id[slot]` | 该槽位最近一个全部结束的批次号。等于自己的 `batch_id`，就说明自己的批次完成了 |
| `timed_out[slot]` | 该槽位当前批次等待超时的标记 |
| cid | NVMe 命令号（command identifier），NVMe 协议里的名字，设备在完成记录里原样带回 |
| `cid_owner[queue][cid]` | 这个 cid 现在属于谁：`{slot, batch_id, entry}` |
| SQE / CQE | 提交队列里的一条命令 / 完成队列里的一条完成记录（NVMe 协议名） |
| `drain_completions(queue)` | 回收：把某个队列上已经到达的完成记录全部处理掉 |
| `finish_entry(slot)` | 一个条目结束：`pending_entries` 减一，减到 0 时写 `finished_batch_id` |
| `sq_enqueue_or_drain` | 把命令放进提交队列；队列满时自己先回收再重试 |
| `wait_batches` | 等待 kernel：等若干批次全部完成 |
| `submitted_event` / `finished_event` | 批次"命令已全部发出" / "全部完成"的 CUDA event |
| `enqueue_wait(handles, stream)` | 新接口：在某条流上排一个"等这些 handle 完成"的等待 |

---

## 1. 结论速览

| 问题 | 回答 |
|---|---|
| 等待者怎么知道一层的命令都完成了 | **不看 cid，看计数。** 每个 batch（DataPath 的一个 op）在显存里有一个剩余计数 `pending_entries`，初值等于本次 launch 的条目数。每个条目恰好让它减一次：提交前就失败的条目由提交线程减，其余条目由回收者在处理它的 CQE 时减。减到 0 的那个线程写 `finished_batch_id = batch_id`。等待者只比较 `finished_batch_id` 是否等于自己的 `batch_id` |
| cid 和 batch 怎么关联 | 每个队列一张常驻显存表 `cid_owner[queue][cid] = {slot, batch_id, entry}`。提交线程拿到 cid 后、写 SQE 之前填表。回收者从 CQE 读出 cid，在它所在队列的表里查到 `{slot, batch_id, entry}`，写这个条目的 status 并给 slot 的计数减一 |
| batch 和 Python handle 怎么关联 | 不变：Python handle → Runtime `IoHandle` → DataPath op → arena 槽位 `slot`。异步模式只给 op 多记一个 `batch_id`。Python 从不接触 cid |
| 谁来回收 | 不用常驻 kernel。回收是一个短例程，由三类已有时机顺带执行：等待 kernel（插在计算流上）、提交 kernel 遇到 SQ 满时、主机 `wait()` 兜底 |
| 为什么不会死锁 | 两条不变式：**等待者只在它要等的提交已经完成后才开始（流上先等 `submitted_event`）**；**任何等待者在等待时自己回收**。于是任何等待都只依赖硬件完成命令，不依赖别的流或别的 kernel 的进度 |
| 收益在哪 | 读写 IO 不再有 kernel 在计算期间驻留，消除持久化 GEMM 的"第二波"。4 盘实测冷 B 的 prefill 步因此多出 163 ms（MoE GEMM 1.89×）；8 盘读窗口减半，预计还剩约 80 ms 可拿回，需先实测确认（§9） |

---

## 2. 现状：kernel 退出即 IO 完成

### 2.1 一个条目的完整路径

当前每个 IO 线程把一条命令从头做到尾（`csrc/data_paths/local_nvme/io/nvme_submit_primitives.cuh:215-263`）：

```
resolve_lba → acquire_queue（全局线程号 % 队列数）
  → get_cid           领 cid（nvm_parallel_queue.h:35-52，cid 空间 0..65535）
  → sq_enqueue        拿 ticket，等槽位空，拷 64B SQE，合并 tail，敲 SQ 门铃（:168-347）
  → cq_poll_bounded   从 CQ head 起扫描，按 cid + phase 找自己的 CQE（:466-500）
  → cq_dequeue        标记 head_mark，抢 head_lock，推进 CQ head、敲 CQ 门铃，
                      同时按 CQE 的 SQ head 字段回收 SQ 槽位（move_head_cq，:82-124）
  → put_cid           还 cid
  → status[idx] = 0/2/3
```

StripedDataPath 的 `fused_submit_kernel_pool`（`fused_submit_kernel.cuh:171-218`）用
`min(条目数, 256)` 个 worker 抢条目执行上面的路径。kernel 后面记一个 CUDA event
（`striped_data_path.cpp:1828`），**这个 event 的语义是"本 batch 全部条目终态"**。上层
全部依赖这一点：

- 主机 `wait(handle)`：`progress_impl_` 查 event，完成后 D2H 拷 status 汇总
  （`striped_data_path.cpp:1900-2031`）。
- 读：feeder 线程提交第 L 层后在读流上记 `read_ready_events[L]`
  （`store.py:1795`），计算流在第 L 层前 `wait_event` 它
  （`worker.py:839` → `store.py:1823-1829`）。
- 写：`save_kv_layer` 在写流上提交，`wait_for_save` 在步末等所有写 handle
  （`worker.py:1095`、`:1165`）。

### 2.2 代价

1. **IO kernel 在整个 IO 期间驻留。** 驻留期间，按整卡 SM 数定 grid 的持久化 GEMM
   （MoE grouped GEMM、FP8 GEMM）被迫多跑一波。4 盘冷 B 的 3328 token prefill 步：读
   kernel 驻留 733 / 1053 ms，计算流 GPU 时间 805 → 968 ms，MoE GEMM 141 → 267 ms
   （详见 `kv-storage-layout-and-io-submission.md` §4.2）。写也一样：A 请求整个
   prefill 期间都有写 kernel 驻留。
2. **超时会卡死队列。** `poll_bounded` 超时返回时既不 `cq_dequeue` 也不 `put_cid`
   （`nvme_submit_primitives.cuh:118-121`）。迟到的 CQE 没人设 `head_mark`，
   `move_head_cq` 走到那一格就停（`nvm_parallel_queue.h:89-91`），这个队列此后的完成
   都推进不了 head，SQ 槽位也不再回收。

---

## 3. 为什么不能直接"提交完就退出"

只把 `poll_bounded` 去掉，会有两个死锁：

1. **CQ head 卡住。** CQ head 只能按顺序推进，每个线程在 `cq_dequeue` 里先标记自己那一
   格，再等 head 越过自己。提交线程退出后，它那条 CQE 没人标记，head 停在那里，后面所有
   完成的线程永远自旋。
2. **SQ 空位不释放。** SQ 槽位只在 CQ head 推进时回收（`move_head_cq` 里给
   `sq->tickets` 加一）。CQ 不前进，下一批提交在 `sq_enqueue` 里等槽位，而负责回收的人
   排在它后面。

根源是"每条完成由发起它的线程认领"。异步化必须换成**由回收者按顺序处理 CQ 上所有已到达
的完成**，完成信息不能再放在发起线程的寄存器里，于是需要一张 cid 到条目的表。

---

## 4. 新模型：提交者、回收者、等待者

### 4.1 数据结构

全部是显存常驻结构，DataPath 初始化时分配，生命周期与队列相同：

```
每个队列（StripedDataPath 有 设备数 × 每设备队列数 个）：
  cid_owner[queue][65536]      uint64，打包 {slot:16, batch_id 低 16 位, entry:32}
                               现有 get_cid 的 cid 空间是 0..65535（queue.h 里 sq_cid
                               也是 65536 项），所以表按 65536 开：每队列 512 KiB，
                               32 个队列共 16 MiB

每个 arena 槽位（已有：StripedArena，槽位数 = 2 × max_in_flight_operations）：
  status[slot][entry]          已有，每条目一个 EntryCompletionStatus
  pending_entries[slot]        新增，uint32
  finished_batch_id[slot]      新增，uint32
  timed_out[slot]              新增，uint32

每个 DataPath：
  all_queues[]                 新增，本 DataPath 所有 QueuePair* 的显存数组，供回收遍历
  next_batch_id                新增（主机侧），每接受一个批次 +1
```

表里只存 `batch_id` 的低 16 位：同一槽位上相邻两个批次的号一定不同，16 位足够区分，§4.6
的防御检查只比较这 16 位。

"批次"就是 **DataPath 的一个 op**：一次 `submit()` 被接受的那些条目，占用一个 arena 槽
位。Python 一层的读通常是一个批次（`_submit_retry` 拆批时是几个，Python 侧对每个 handle
都等）。

### 4.2 提交 kernel（新增，只提交）

worker 池结构与现在相同。kernel 参数里带上本批次的 `slot` 和 `batch_id`，单个条目
`entry` 变成：

```
if resolve_lba 失败:
    status[slot][entry] = 1
    finish_entry(slot)
    继续下一个条目
queue = acquire_queue()
cid   = get_cid(queue)
cid_owner[queue][cid] = {slot, batch_id, entry}   // 必须在命令可见之前写入
__threadfence()                                     // 表项先于命令和门铃对回收者可见
sq_enqueue_or_drain(queue, 命令)                    // 队列满时自己回收，不空等
// 不等完成，直接做下一个条目
```

`finish_entry(slot)`：`atomicSub(&pending_entries[slot], 1)` 返回 1 时（自己是最后一个），
先 `__threadfence()`，再写 `finished_batch_id[slot] = batch_id`（release 语义）。

`sq_enqueue_or_drain` 是新函数，逻辑与现有 `sq_enqueue` 相同，只是在等 ticket 和等 tail
的两个自旋循环里，每轮先对**本队列**调一次 `drain_completions(queue)`。这样提交队列满
时，提交线程自己推进完成队列、释放提交队列的空位，而不是指望别人。

`pending_entries[slot]` 由主机在 launch 前随 entries 一起 H2D 写入（同一条流、同一次拷
贝），初值为本批次的条目数。拷贝在 kernel 开始前已经完成，所以任何回收者处理到这个批次的
完成记录时，计数一定已经初始化好（因果链：H2D → kernel 写表和命令 → 设备取命令 → 完成
记录）。

### 4.3 回收例程 `drain_completions(queue)`（新增）

```
if (!try_lock(queue.drain_lock)) return;     // 别人在收：直接返回，从不在锁外等
n = 0
while n < 本次上限 且 完成队列[head + n] 的 phase 位表明它是新到的:
    cpl = 完成队列[head + n]
    {slot, batch_id, entry} = cid_owner[queue][cpl.cid]
    status[slot][entry] = 0（成功）或 3（NVMe 报错）
    put_cid(queue, cpl.cid)
    finish_entry(slot)                        // 可能写 finished_batch_id[slot]
    n++
if n > 0:
    按最后一条完成记录里的"提交队列 head"字段释放提交队列空位（逐格 tickets +1）
    完成队列 head += n，敲完成队列门铃
unlock(queue.drain_lock)
```

要点：

- **锁内只处理已经到达的完成，从不等待。** 持锁时间正比于本次处理的 CQE 数，别的回收者
  拿不到锁就返回，不会互相阻塞。
- 不再有"按线程顺序认领"，CQE 不管属于哪一层、哪个 op、读还是写，都一起处理。
- 释放提交队列空位的写法与现有 `move_head_cq` 相同（tickets 每格 +1，与 `sq_enqueue`
  的轮次约定一致），只是写在新函数里。
- **同一个队列不能新旧两种模式混用**：旧模式的线程按 cid 认领自己的 CQE，回收者会把它
  "抢走"。所以模式按 DataPath 实例选定，一个 DataPath 的所有 launch 用同一种模式。

### 4.4 等待 kernel（新增）

一个 block、32 线程：

```
wait_batches(要等的批次列表 [{slot, batch_id}, ...], all_queues[], 轮询上限):
    loop:
        第 k 号线程负责 all_queues[k], all_queues[k + 32], ...：各调一次 drain_completions
        if 列表里每个批次都满足 finished_batch_id[slot] == batch_id: return   // 正常完成
        if 超过 deadline 且尚未报告: 给未完成批次置 timed_out[slot] = 1，继续回收等待
        nanosleep（退避）
```

它插在需要数据的那条流上，前面**先 `cudaStreamWaitEvent(该批次的 submitted_event)`**。
`submitted_event` 就是现在提交 kernel 后面记的那个 event；异步模式下它的含义从"全部完成"
变成"命令已全部写进提交队列并敲过门铃"。等待 kernel 后面再记一个 `finished_event`，主机
`wait()` 查它。

### 4.5 问题一：等待者怎么知道"当前层提交的 cid 都完成了"

**等待者不关心 cid，它等的是批次的计数归零。**

- 一层的读 = 一个（或几个）批次。每个批次有自己的 `pending_entries[slot]`，初值 = 本批次
  的条目数。
- 每个条目恰好让计数减一次，且只在它**结束**时减：
  - 提交前失败（`resolve_lba` 不通过、故障注入）：提交线程立即减；
  - 命令已发出：只在回收者处理它的完成记录时减，成功和 NVMe 报错都算结束；
  - 等待超时：**不减**（命令可能仍在设备上），见 §6。
- 减到 0 的线程先 `__threadfence()`，再写 `finished_batch_id[slot] = batch_id`。所以等待
  者看到 `finished_batch_id[slot] == batch_id` 时，该批次所有条目的 status 都已写好。
- 等待者自己也在回收，所以它等的完成一定会被处理，不依赖别人。

为什么不让等待者枚举 cid：cid 是提交线程在运行时从各自队列动态领取的，主机和等待者都不
知道这一层用了哪些队列、哪些 cid；枚举只能扫批次的全部条目，代价与条目数成正比，而且需要
提交前把 status 置成"未完成"（现在为性能去掉了这次 memset）。计数是 O(1) 的。

### 4.6 问题二：cid table 和 batch IO、handle 的关联

```
Python handle（整数 id）
  └─ Runtime IoHandle                                        已有
       └─ DataPath op { slot, batch_id, submitted_event, finished_event }
                                                             已有 + batch_id / finished_event
显存（槽位 slot）
  entries[slot][0..n)   status[slot][0..n)   pending_entries[slot]   finished_batch_id[slot]

提交线程处理 entries[slot][entry]：
  cid = get_cid(queue)
  cid_owner[queue][cid] = {slot, batch_id, entry}      ← 记下"这条命令属于谁"
  命令(cid) → 提交队列[queue] → 设备

回收者处理完成队列[queue] 上的一条完成记录(cid)：
  {slot, batch_id, entry} = cid_owner[queue][cid]      ← 查出"这条完成属于谁"
  status[slot][entry] = 结果
  pending_entries[slot] -= 1；归零时 finished_batch_id[slot] = batch_id

等待 kernel / 主机：只看 finished_batch_id[slot] == batch_id
```

- **一条命令在任一时刻由 `(queue, cid)` 唯一标识**，表项在命令发出前写入、在它的 CQE 被处理
  后随 `put_cid` 失效（cid 只有归还后才会被重新领取，表项在那之前不会被覆盖）。
- **一个条目对应一条命令**（现在的提交路径在主机侧已把请求按 MDTS/extent 切成条目，
  一条目一命令）。
- `batch_id` 防止槽位复用后把旧批次的完成算到新批次上：表项里的 `batch_id` 与槽位当前批
  次的号不一致时，只归还 cid、计入错误计数，不写 status、不减计数。正常情况下超时槽位会被
  隔离（§6），这条分支不应出现，它是防御。
- Python 侧接口只多一个"在某条流上等某个 handle 完成"的调用，handle 照旧是一个整数 id。

---

## 5. 无死锁与正确性论证

| 风险 | 结论与依据 |
|---|---|
| 等待者等一个永远不会发出的提交（跨流环） | 读流有时会等计算流（`core.py:2163` 的 `protect_read_io`），写流等计算流的 gather fence（`core.py:1876`）。等待 kernel 前面先等**它所等批次的 `submitted_event`**，所以它开始执行时，该批次的命令已经全部进了提交队列。此后它的完成只依赖设备和它自己的回收，与任何流的进度无关，不会成环 |
| SQ 满 | 提交线程在 `sq_enqueue_or_drain` 里自己回收本队列。只要设备在完成命令，槽位就会释放 |
| CQ 溢出 | 在途命令数 ≤ SQ 深度 − 1（槽位只在其 CQE 被处理后释放），CQ 深度 ≥ SQ 深度，CQ 永远不会满 |
| 回收者互相阻塞 | 回收锁是 try-lock，锁内不等待；拿不到锁的人检查完自己的条件就退避重试 |
| 回收者处理一条 CQE 时表项还没写好 | 提交线程先写表、`__threadfence()`，再写 SQE 并经 `sq_enqueue` 敲门铃；设备只能在门铃之后取命令、之后才写 CQE。回收者读表用 acquire 语义 |
| 计算看到未落地的数据 | 数据与 CQE 都由 SSD 经 PCIe 写入同一块 GPU 显存，同一请求方的 posted write 保序，CQE 可见时数据已到。消费数据的 kernel 在等待 kernel 之后按流序启动 |
| 等待 kernel 挡住计算 | 它在计算流上与 GEMM 串行，不会同时驻留；数据已到时几微秒退出，没到时计算本来也要等 |
| CUDA Graph | connector 声明 `requires_piecewise_for_cudagraph`（`connector.py:309`），逐层等待在 piecewise 片段之间 eager 执行，与现在的 `wait_event` 同一位置，不进图 |

---

## 6. 超时与故障

- 等待 kernel 超过 deadline：置 `timed_out[slot] = 1`，但继续回收直到实际完成。
  不能提前退出让消费流读未落地的数据。主机 `wait()` 有独立观察超时，可返回 TIMEOUT；
  观察超时不取消设备命令。
- **超时批次资源保留**：状态保持 IN_FLIGHT，不释放 op、arena、PRP、内存注册或目标租约。
  迟到 CQE 正常回收；`finished_batch_id` 匹配且所有已登记等待流的退休事件完成后，
  才发布 FAILED/TIMEOUT 并释放资源。异步路径不再走提前回收槽位的 `release_with_timeout_leak`。
- `shutdown(timeout_ns)` 超时保留资源，拒绝新提交，可再调用以完成 drain。析构不能返回并
  自动销毁还在使用的成员资源，因此可能等待；设备永久挂死必须先由运维安全恢复，不能
  用 free 或第二次信号冒充 NVMe Abort。
- Runtime 即使一个子操作失败，也等所有其它子操作安全退休；query/progress 错误不能
  证明 DMA 取消。已经完成但释放失败时保留结果和引用，允许重试。
- 命令永远不完成（设备挂死）：cid 和槽位一直被占，与现在一样需要队列重置；本设计不处理
  NVMe Abort / 控制器复位。
- 主机兜底回收：如果某个批次没有任何设备侧等待者（例如 abort 路径、只在主机上
  `wait()` 的路径），`progress_impl_` 的异步版本在 `wait()` 超过短暂宽限期仍未完成时，
  在 DataPath 内部的回收流上发一次等待 kernel。主机正在阻塞等待，说明已经到了同步点，
  这时有一个短 kernel 驻留不影响计算。

---

## 7. 主机与 Python 接口（全部新增）

**DataPath（新增方法，异步模式实例才有）**

- `submit()` 不变（SPI 不变），返回的 op 在异步模式下多记 `batch_id` 和 `finished_event`。
- `enqueue_wait(op, stream)`：在 `stream` 上插 `cudaStreamWaitEvent(op.submitted_event)` +
  `wait_batches`，并在其后记 `op.finished_event`。可一次传多个批次。
- `progress` 的异步版本：检查 `finished_batch_id` 的主机可见镜像与所有等待流的退休事件；
  两者同时完成才终态，必要时触发 §6 的兜底回收。完成后的 status D2H 汇总与同步相同。

**Runtime / pybind（新增）**

- `Runtime::enqueue_wait(handles, stream)` → 找到各 handle 的 op，按 DataPath 分组转发。

**Python（新增方法，worker 两处按开关二选一）**

- `store.enqueue_wait(handles, stream=None)`：默认当前计算流。
- 读：`wait_for_layer_load` 里，开关打开时用 `enqueue_wait(本层 handles)` 代替
  `wait_compute_event(read_ready_events[L])`。
- 写：`wait_for_save` 里，开关打开时先 `enqueue_wait(本步写 handles)`，再走现有的
  `wait_result`（此时已完成，只做汇总）。

---

## 8. 接入方式：不改动现有函数

现有的 `get_cid` / `sq_enqueue` / `cq_poll_bounded` / `cq_dequeue` / `submit_read_one`
/ `fused_submit_kernel_pool` / `StripedDataPath::submit_impl_` 等全部保持不动，同步模式
行为逐字节不变。新增内容：

| 新增 | 位置 |
|---|---|
| `sq_enqueue_or_drain`、`drain_completions`、`finish_entry`、`issue_nvme_cmd_no_wait`、`submit_read_no_wait / submit_write_no_wait` | 新头文件 `csrc/data_paths/local_nvme/io/nvme_async_primitives.cuh` |
| 只提交 kernel、`wait_batches` kernel 及 launch 函数 | 新文件 `csrc/data_paths/striped_local_nvme/async_submit_kernel.{cuh,cu}` |
| `cid_owner`、`pending_entries` / `finished_batch_id` / `timed_out`、`all_queues` 的分配与生命周期 | 新文件 `csrc/data_paths/striped_local_nvme/async_completion.{h,cpp}` |
| 异步模式的 DataPath | 见下 |
| `Runtime::enqueue_wait`、pybind 导出、`store.enqueue_wait` | 新增函数 |
| 硬件契约测试 | 新目录 `tests/data_paths/async_completion/` |

**必须触碰的接入点（都是新增分支，默认关闭）**：preset / 配置里一个新开关
`completion: async`（默认 `sync`），以及 worker 两处按开关选择等待方式。

**DataPath 怎么承载异步模式**，两种做法，需要定：

- **A. 派生类**：`StripedAsyncDataPath : StripedDataPath`，复用初始化、队列建立、arena、
  PRP、resolver payload，只覆盖 launch 和完成判定。需要把 `StripedDataPath` 的若干成员
  从 `private` 改为 `protected`、两三个函数改为 `virtual`。这是对现有类的改动，但不改
  任何函数体、不改同步模式行为。改动最小，**推荐**。
- **B. 完整复制**：新类从 `StripedDataPath` 复制约 2000 行再改。完全不碰现有文件，但两份
  代码会分叉，后续每个修复都要改两遍。

LocalNvmeDataPath（单盘）暂不做：生产路径是 StripedDataPath，验证通过后再按同样方式加。

---

## 9. 收益、代价与验证计划

**收益（估计，待实测）**

- 冷 B（8 盘，64K 复用 95%）：1.837 s，HBM 1.637 s。4 盘时 IO 驻留导致计算多 163 ms；
  8 盘读窗口约减半，估计可拿回约 80 ms。
- A：整个 prefill 都有写 kernel 驻留，A 比 HBM 慢的约 230 ms 里有一部分来自这里。
- 在线大读（≥ 512 块/步）受盘带宽限制，计算不在关键路径上，收益小。

**代价**

- 每层多一次等待 kernel launch（约 5–10 µs），80 层每步约 0.4–0.8 ms。
- 只提交 kernel 仍会短暂驻留（每层几十微秒），恰好与某个 GEMM 重叠时那个 GEMM 多一波。
- 每 DataPath 约 16 MiB 显存（`cid_owner`）。

**验证步骤**

0. **先测上限**：8 盘冷 B 跑一次 nsys，对比 HBM 同一步的计算流 GPU 时间。差值是能拿回的
   上限；如果只剩几十毫秒，重新评估是否值得做。
1. 原语单测：单卡单盘，提交 N 条后退出、等待 kernel 收齐，逐字节校验；SQ 满（N ≫ 队列
   深度）、多流并发、混合读写。
2. 注入超时：等待 kernel 超时 → 槽位隔离 → 迟到 CQE 被正常回收 → 槽位释放 → 队列继续可用。
3. 两阵列 RAID0 逐字节对照（`tutti_raid0_local_nvme_contract_test`）与 chrdev 身份测试
   （`tutti_snvme_identity_contract_test`）在异步模式下复跑；Python 全套。
4. 端到端：冷 / 热 B、A，与同步模式和 HBM 同日对照；nsys 确认计算期间无 IO kernel 驻留。

---

## 10. 未决问题

1. §8 的 A / B 选哪一种。
2. `finished_batch_id` 的主机可见方式：映射的 pinned 内存（主机零拷贝轮询，需 system scope 的
   release 写）还是每次 D2H。前者主机轮询便宜，后者实现简单。
3. 写入的回收时机：A 请求没有读，层间的写完成只能由后续层的提交 kernel（SQ 满时）和步末
   `wait_for_save` 回收。是否让每个提交 kernel 开始时顺带回收一次（成本是提交 kernel
   略长），需要实测决定。
4. cid_owner 大小：现有 `get_cid` 的 cid 空间是 65536，所以表按 65536 开。若新增一个把
   cid 限制在 SQ 深度内的分配函数，表可缩到 SQ 深度（约 1–4K 项）。

---

## 11. 实现与实测（2026-10-09）

### 11.1 实现与本文设计的差异

| 设计稿 | 实际实现 |
|---|---|
| 开关 `completion: async`（配置项） | 环境变量 `TUTTI_ASYNC_COMPLETION=1`，YAML 工厂与 preset 两条构建路径都读它；Python 侧读同一个变量 |
| `complete_on` / `enqueue_wait` 接口 | `tutti::AsyncCompletion::enqueue_wait`（新 SPI 头 `spi/async_completion.h`，DataPath 可选实现）→ `StorageRuntime::enqueue_wait` → pybind `runtime.enqueue_wait(handles, stream)` → `store.enqueue_compute_wait(completions)` |
| 新文件 `async_submit_kernel.cu` | `async_submit_kernel_impl.cuh`，由 `fused_submit_kernel.cu` 末尾 include：原头文件在头里定义 kernel，第二个翻译单元会在设备链接时重复定义 |
| 主机看 `finished_event` | 主机先读 `finished_batch_id` 镜像，再检查每个已登记等待流的退休事件；二者同时满足才终态 |
| §6 主机兜底回收 | 已实现：提交完成 100 ms 后仍无人排等待的批次，在 DataPath 内部流上补一个等待 kernel（每进程只打印一次日志）。只在主机侧等待的调用方（预热读、契约测试）走这条 |

对现有代码的改动（方案 A）：`StripedDataPath` 两处 `private` → `protected`；新增 virtual
`launch_io_kernel_`，`submit_impl_` 里原先直接调用 `launch_fused_submit` 的一处改为调它
（同步模式下仍是原调用、原参数）。其余已有函数未改。

Python 接入点：`worker.wait_for_layer_load`（异步模式下计算流等本层读批次完成，替代
`wait_event(fence)`）、`KVEngine.wait_idle` 开头（步末在计算流上为全部在途批次排等待，
写入由此回收）。

### 11.2 正确性

| 测试 | 结果 |
|---|---|
| `tutti_async_completion_bench --verify`，GPU0→kv0、GPU4→md0，同步 / 异步各一次：写 2 GiB 图样 → 读回 → GPU 比对 | 全部 0 不一致 |
| 提交队列打满：每盘 1 个队列，单批 16384 条（每队列 4096 条） | 同步 / 异步均 0 不一致 |
| `tutti_raid0_local_nvme_contract_test`（异步模式，走主机兜底回收） | 18/18 |
| Python 455、非硬件 ctest 31、chrdev 身份 31/31、RAID0 18/18（同步模式，确认默认路径不变） | 全过 |
| 离线 e2e 输出 token | 同步 / 异步完全一致 |
| 当时未测：超时与迟到完成退休 | 2026-10-10 已补 GPU-only 合成故障测试，见 §12；真实 SSD 故障注入仍未做 |

### 11.3 性能

**C++ 基准**（单卡 GPU0、kv0 四盘，80 层 × 每层 32 MiB 读全部预先提交，计算流逐层等
读后跑 1 个按 SM 数定 grid、每 block 用满共享内存的持久化 kernel，约 2.5 ms/层）：

| | 只算 | 算 + IO：计算耗时 | IO 带宽 |
|---|---:|---:|---:|
| 同步 | 196.7 ms | 258–264 ms（**+31~34%**） | 9.0–9.2 GiB/s |
| 异步 | 196.7 ms | 196.6 ms（**+0.0%**） | 11.6 GiB/s |

cuBLAS 普通 GEMM（n=8192，非持久化）两种模式都不变慢：问题只出在按 SM 数定 grid 的持久化 kernel 上。

**离线 e2e**（HY3 TP8，64K token，复用 95%，两阵列 8 盘，3 轮稳态均值，同日对照）：

| | 同步 | 异步 | 差 |
|---|---:|---:|---:|
| A（从零算并写盘） | 10.260 s | 10.147 s | **−113 ms** |
| 冷池 B（95% 读盘 + 5% 新写） | 1.843 s | 1.796 s | **−47 ms** |

**nsys**（`reports/8gpu-tutti-65536-reuse95-{sync,async}-nsys-20261009-*.nsys-rep`，
sqlite 在 `tmp-export/{sync,async}-1009.sqlite`）：B 的 3328 token prefill 步，rank 0：

| | 同步 | 异步 |
|---|---:|---:|
| 步长 | 925.9 ms | 907.7 ms |
| 读流 IO kernel 驻留 | 384.6 ms | 21.3 ms（160 次只提交 kernel） |
| 写流 IO kernel 驻留 | 36.3 ms | 4.7 ms |
| MoE grouped GEMM（cutlass） | 174.9 ms | **141.4 ms**（HBM 参考 141 ms） |
| FP8 GEMM | 39.1 ms | 29.2 ms（HBM 参考 27 ms） |
| 计算流上的等待 kernel | — | 85 次，共 4.8 ms |

驻留导致的"第二波"已经消除：MoE / FP8 GEMM 回到 HBM 水平。剩余与 HBM 的差距（冷 B
1.796 vs 1.637 s）不再来自 SM 争用：层 0 等首批读约 10 ms，all-reduce 多 14 ms（跨 rank
到达时间不齐），以及读本身受盘带宽限制（每步约 19 GiB / 8 盘）。

**256K 复用**（prompt 262136 token = 模型上限 262144 − 生成的 8 个，复用 95%，其余同上）：

| | HBM 前缀缓存 | 同步 | 异步 | 异步 − 同步 |
|---|---:|---:|---:|---:|
| A（稳态均值） | 100.791 s | 101.885 s | 101.489 s | −0.40 s |
| 冷池 B（稳态均值） | 9.731 s | 11.022 s | **10.094 s** | **−0.93 s（−8.4%）** |
| B 三轮波动 | ±0.004 s | 10.83–11.21 s | 10.09–10.10 s | |

与 HBM 的差距从 1.29 s 缩到 0.36 s（缩小 72%）。输出 token 两种模式完全一致。

nsys（`reports/8gpu-tutti-262136-reuse95-{sync,async}-256k-nsys-20261009-*.nsys-rep`，
sqlite `tmp-export/{sync,async}-256k.sqlite`）：B 的 prefill 步（13304 token，rank 0）
10175 → 9342 ms（−833 ms）。

| | 同步 | 异步 |
|---|---:|---:|
| 读 IO kernel 驻留 | 1208.4 ms | 90.4 ms |
| 写 IO kernel 驻留 | 141.9 ms | 6.4 ms |
| all-reduce（rank 0） | 1252.2 ms | **449.2 ms** |
| MoE grouped GEMM | 432.3 ms | 388.0 ms |
| FP8 GEMM | 133.3 ms | 102.8 ms |
| FlashAttention（占步长 86%） | 8000.2 ms | 8000.6 ms |

256K 下收益主要不在 GEMM，而在 **all-reduce**。分析脚本
`/mnt/nvme4/tutti-profile/raid0/persist/rank_skew.py` 统计每个 all-reduce 上 8 个 rank 的
到达时间差（最晚 − 最早）：同步合计 1219 ms、最大 183 ms；异步 404 ms、最大 58 ms。
同步模式下哪个 rank 的 GEMM 碰上自己的 IO kernel 驻留、被挤出第二波，取决于该 rank 的读
何时到达，8 个 rank 各不相同；TP 每层两次 all-reduce 都要等最慢的那个，于是每个 rank 的
随机变慢被放大成全员等待。异步模式消除了这一随机项，B 的三轮波动也从 ±190 ms 收窄到
±3 ms。

## 12. 生命周期修正与故障验证（2026-10-10）

本节是当前代码的安全契约；§11 的性能数字来自此前版本，不代表本次修正已重新做真实性能验收。

### 完成与资源退休分开确认

1. 超过 deadline 只置 `timed_out`；等待 kernel 继续回收，不让消费流在 IO 未完成时前进。
2. `finished_batch_id` 匹配表示命令已完成；还须检查所有已登记等待流的 CUDA event 已退休。
3. 两个条件同时满足后才发布终态。超时批次最终为 FAILED/TIMEOUT，之后可正常归还 arena。
4. 在此之前 op、PRP、arena、内存注册与目标租约均保持引用；`release`/close/unregister 不能
   以观察错误代替硬件取消。异步路径不调用提前复用槽位的 timeout-leak 释放。
5. 每批 H2D 的源状态放在稳定的主机容器中，不再引用函数返回后的栈。

### 关闭与错误重试

- `shutdown` 先禁止新增提交，给在飞批次排回收等待；主机期限到达返回 TIMEOUT，保留完整
  资源，下次可继续 drain。时间预算是轮询的合作式限制，不能中断一个已经开始的 CUDA API。
- 所有批次真正完成且所有等待 event 已退休后，才释放队列、arena 和 completion tables。
- Runtime 等全部异步子操作结束，不因一个失败或 query/progress 错误提早归还引用。
- 子操作 terminal snapshot 保留到释放成功；`release_retired(op, retired)` 区分实际释放
  与后续 guard 恢复错误，已发生的 release 不重放。
- submit event 记录失败时，stream 同步只证明命令已提交；不能证明 SSD IO 完成。独立跟踪
  submit 退休状态。等待 event 记录失败时保留等待引用，允许原流补录退休 event。
- `wait/query` 在放锁驱动进度后重新验证原 handle，避免另一个线程复用槽位后返回新批结果。
- 析构不能靠 return 阻止成员析构；异步 owning Runtime/DataPath 必须保留对象直到安全排空。
  因而永久挂死的设备可能导致析构等待。无 NVMe Abort/reset 确认时，不以强行 free 代替恢复。

### 本轮测试

- Python index 3 个反例：同批驻留 key 不能被驱逐、容量不足无副作用、批外 pin 保持保护。
- Runtime 软件 fake：query/progress 错误、失败子操作旁仍有在飞子操作、释放前失败、实际释放
  后 restore 失败、Runtime 槽位复用清理错误、已受理 op 的状态数组异常。
- `tutti_async_completion_contract_test`（`hardware;gpu_only` 标签，不打开 NVMe）：
  - 显存合成 CQ 验证 `(queue, cid)` 到两个批次的映射、完成乱序、phase 回卷与 CID/SQ 回收；
  - gated producer 制造迟到完成，超时后状态与 arena 不可释放；
  - 第一条等待结束但第二条流仍被 gate 阻塞，批次继续在飞；
  - shutdown 超时后仍保留表/arena，并拒绝新请求；解除 gate 后可重试关闭；
  - 模拟缺失 submit/等待事件并检查安全退休；一次等待 20 个批次跨 16 项分组；零进度预算不推进。
- 483 个 Python 测试、31 个非硬件 C++ 用例通过；GPU-only 契约单独通过。
- compute-sanitizer memcheck 尝试超过 90 秒，只有启动 banner，没有结果；**不能记为通过**。

未做：本轮修正后的真实 NVMe/LLM/性能回归、真实驱动错误/永久设备故障与 reset 认证。
默认仍使用同步完成模式，不改变部署中的 daemon 或既有盘数据。
