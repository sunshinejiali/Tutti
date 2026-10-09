#!/usr/bin/env bash
# Hand an md RAID0 over to its own tutti_daemon, and give it back to Linux.
#
#   tutti-md.sh <kv0|md0> up      [--confirm NAME]  kernel nvme + md -> snvme + daemon (supervised)
#   tutti-md.sh <kv0|md0> down    [--confirm NAME]  stop the daemon; the supervisor restores Linux
#   tutti-md.sh <kv0|md0> restore [--confirm NAME]  (idempotent) back to kernel nvme + md + mount
#   tutti-md.sh <kv0|md0> status
#   tutti-md.sh <kv0|md0> level                     prints normal|protected
#
# Levels (per array, in the table below):
#   normal     the array holds only data Tutti can lose (KV pools).
#   protected  the array holds host data (models, datasets, docker). Every
#              driver switch done by hand (up / down / restore) must name the
#              array again with --confirm NAME, and is refused -- never waited
#              out -- while anything still uses it: the array must be clean
#              with all members, and nothing may hold its mount, its block
#              devices or (going back) its controllers. The supervisor's
#              automatic fallback after a daemon exit needs no confirmation.
#
# Every level: the controllers are never released while a Tutti client (e.g.
# a vLLM worker) still drives them -- that can hang or crash the client and
# its GPU. `down` refuses; the automatic fallback waits for the clients.
#
# Safety model (never release a controller under a mounted filesystem):
#   * `up` runs the daemon under a supervisor. Whenever the daemon exits --
#     clean stop, startup failure, crash, SIGKILL -- the supervisor runs
#     `restore`: wait for Tutti clients, unmount (waiting for users), stop the
#     array, rebind the members to the kernel nvme driver, let md assemble the
#     SAME superblocks, mount it again (md0: its fstab entry), restart docker
#     if `up` stopped it.
#   * `up` only succeeds after a health check (daemon serving, the array
#     mounted on snvme members, top-level listing identical to before); on any
#     failure it stops the daemon and the supervisor rolls back.
#   * Nothing on disk is rewritten: no mkfs, no superblock changes.
#   * Logs go to /var/log, never onto a managed mount.
set -uo pipefail

REPO="${TUTTI_MD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DAEMON="${TUTTI_MD_DAEMON:-$REPO/build/bin/tutti_daemon}"   # override: failure drills only
NAME="${1:-}"; CMD="${2:-}"
if [ $# -ge 2 ]; then shift 2; else shift $#; fi
CONFIRM=""
while [ $# -gt 0 ]; do
    case "$1" in
        --confirm) CONFIRM="${2:-}"; if [ $# -ge 2 ]; then shift 2; else shift; fi ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

case "$NAME" in
# PCIS: the members in the order boot enumerated them (pci_topology_check.sh:
# d2,df,86,c5 -> nvme0-3; 08,4b,57,63 -> nvme4-7). restore binds in this
# order; the kernel hands out the lowest free nvmeN, so restoring md0 before
# kv0 reproduces the boot names. Names are cosmetic: every check here uses
# PCI addresses and the array UUID, never nvmeN/snvmeN/ssnvmeN numbers.
# LEVEL: see "Levels" above.
kv0) CONFIG="$REPO/config/local/tutti_daemon_kv0.yaml"; PORT=50051; LEVEL=normal
     PCIS="0000:08:00.0 0000:4b:00.0 0000:57:00.0 0000:63:00.0"
     UUID="83779789:3087c10e:59288f94:b9efc29c"; MOUNT=/mnt/tutti_md0; DOCKER=0 ;;
md0) CONFIG="$REPO/config/local/tutti_daemon_md0.yaml"; PORT=50052; LEVEL=protected
     PCIS="0000:d2:00.0 0000:df:00.0 0000:86:00.0 0000:c5:00.0"
     UUID="37fc2412:b826695c:bc0a387a:ff29403e"; MOUNT=/mnt/nvme4; DOCKER=1 ;;
*) usage; exit 2 ;;
esac
[ "$CMD" = level ] && { echo "$LEVEL"; exit 0; }

STATE=/run/tutti-md/$NAME
LOG=/var/log/tutti-md-$NAME.log
DLOG=/var/log/tutti_daemon_$NAME.log
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }
mkdir -p "$STATE"

log() { echo "$(date '+%F %T') [$NAME] $*" | tee -a "$LOG" >&2; }
die() { log "ERROR: $*"; exit 1; }

md_dev() {  # /dev/mdN for our UUID, "" if not assembled
    local l=/dev/disk/by-id/md-uuid-$UUID
    [ -e "$l" ] && readlink -f "$l" || true
}
md_members() {  # member kernel names of the assembled array
    local d; d=$(md_dev); [ -n "$d" ] && ls "/sys/block/$(basename "$d")/slaves" 2>/dev/null | sort | tr '\n' ' '
}
driver_of() {
    local l; l=$(readlink "/sys/bus/pci/devices/$1/driver" 2>/dev/null)
    if [ -n "$l" ]; then basename "$l"; else echo none; fi
}
alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }
daemon_pid() { cat "$STATE/daemon.pid" 2>/dev/null; }
supervisor_pid() { cat "$STATE/supervisor.pid" 2>/dev/null; }
mounted_src() { findmnt -no SOURCE "$MOUNT" 2>/dev/null; }

pids_of() { tr -s ' \t' '\n\n' | grep -E '^[0-9]+$' | sort -un | tr '\n' ' '; }
describe() { local p; for p in "$@"; do echo -n "$p($(cat "/proc/$p/comm" 2>/dev/null)) "; done; }
without_daemon() { local d; d=$(daemon_pid); tr ' ' '\n' | grep -vx "${d:-x}" | grep . | tr '\n' ' '; }

# The members' character devices: exactly those the daemon holds open (its own
# controllers), recorded at `up` for use after the daemon has died.
member_chrdevs() {
    local pid; pid=$(daemon_pid)
    if alive "$pid"; then
        ls -l "/proc/$pid/fd" 2>/dev/null | grep -oE '/dev/ssnvme[0-9]+' | sort -u | tr '\n' ' '
    else
        cat "$STATE/chrdevs" 2>/dev/null
    fi
}
chrdev_clients() {  # processes other than the daemon driving the members
    local devs; devs=$(member_chrdevs); [ -n "$devs" ] || return 0
    # shellcheck disable=SC2086
    fuser $devs 2>/dev/null | pids_of | without_daemon
}
mount_users() { fuser -m "$MOUNT" 2>/dev/null | pids_of | without_daemon; }
device_users() {  # processes with the array or a member block device open
    local md m devs; md=$(md_dev); [ -n "$md" ] || return 0
    devs="$md"; for m in $(md_members); do devs="$devs /dev/$m"; done
    # shellcheck disable=SC2086
    fuser $devs 2>/dev/null | pids_of
}
members_on_kernel() { local p; for p in $PCIS; do [ "$(driver_of "$p")" = nvme ] || return 1; done; }
array_clean() {  # $1=/dev/mdN: running, every member present
    local b; b=$(basename "$1")
    grep -qE '^(clean|active|active-idle)$' "/sys/block/$b/md/array_state" 2>/dev/null &&
        [ "$(ls "/sys/block/$b/slaves" 2>/dev/null | wc -l)" = "$(echo $PCIS | wc -w)" ]
}

require_confirm() {  # $1=command; protected arrays only
    [ "$LEVEL" = protected ] || return 0
    [ "$CONFIRM" = "$NAME" ] || die "$NAME is protected ($MOUNT holds host data) and '$1' switches its drivers. Make sure nothing uses it, then re-run with --confirm $NAME"
}

# Wait (up to 30 min) until no Tutti client drives the members.
wait_no_clients() {
    local c i
    for i in $(seq 1 1800); do
        c=$(chrdev_clients); [ -z "$c" ] && return 0
        [ $((i % 30)) = 1 ] && log "waiting for Tutti clients of $NAME's controllers to exit: $(describe $c)"
        sleep 1
    done
    return 1
}

umount_waiting() {  # unmount MOUNT, waiting (up to 30 min) for its users
    local i
    for i in $(seq 1 1800); do
        findmnt "$MOUNT" >/dev/null || return 0
        umount "$MOUNT" 2>/dev/null && return 0
        [ $((i % 15)) = 1 ] && log "waiting for users of $MOUNT: $(fuser -m "$MOUNT" 2>/dev/null | tr -s ' ')"
        sleep 1
    done
    return 1
}

# ---------------------------------------------------------------------------
restore() {
    log "restore: start (drivers: $(for p in $PCIS; do echo -n "$(driver_of $p) "; done))"
    if ! members_on_kernel; then
        wait_no_clients || die "restore: Tutti clients still drive $NAME after 30 min; controllers stay up (stop them, then: $0 $NAME restore)"
    fi
    local pid; pid=$(daemon_pid)
    if alive "$pid"; then
        log "restore: stopping daemon $pid"
        # One SIGTERM only: the daemon then unmounts, stops the array and
        # waits as long as the mount is busy. A second signal would make it
        # release the controllers under a mounted filesystem (XFS shuts down).
        kill -TERM "$pid"
        local i=0
        while alive "$pid"; do
            [ $((i % 30)) = 0 ] && log "restore: daemon $pid still unmounting; users of $MOUNT: $(fuser -m "$MOUNT" 2>/dev/null | tr -s ' ')"
            i=$((i + 1)); sleep 1
        done
    fi
    rm -f "$STATE/daemon.pid"

    # Array still on snvme members (daemon crashed / force-exited): take it down.
    local md; md=$(md_dev)
    if [ -n "$md" ] && md_members | grep -q snvme; then
        if [ "$(mounted_src)" = "$md" ]; then
            umount_waiting || die "restore: $MOUNT still busy after 30 min; members stay on snvme (array is still usable)"
        fi
        mdadm --stop "$md" >>"$LOG" 2>&1 || die "restore: mdadm --stop $md failed"
    fi

    # Members back to the kernel nvme driver.
    local p d
    for p in $PCIS; do
        d=$(driver_of "$p")
        if [ "$d" != nvme ]; then
            [ "$d" != none ] && echo "$p" > "/sys/bus/pci/drivers/$d/unbind" 2>>"$LOG"
            for _ in 1 2 3; do echo "$p" > /sys/bus/pci/drivers/nvme/bind 2>>"$LOG" && break; sleep 2; done
        fi
        [ "$(driver_of "$p")" = nvme ] || die "restore: $p did not bind to nvme (driver=$(driver_of "$p"))"
    done

    # md assembles the same superblocks (udev incremental, else explicit).
    udevadm settle --timeout=30
    for _ in $(seq 1 30); do md=$(md_dev); [ -n "$md" ] && grep -q "active" "/sys/block/$(basename "$md")/md/array_state" 2>/dev/null && break; sleep 1; done
    md=$(md_dev)
    if [ -z "$md" ] || ! grep -qE "clean|active" "/sys/block/$(basename "$md")/md/array_state" 2>/dev/null; then
        [ -n "$md" ] && mdadm --stop "$md" >>"$LOG" 2>&1
        mdadm --assemble --scan --uuid="$UUID" >>"$LOG" 2>&1
        udevadm settle --timeout=30; md=$(md_dev)
    fi
    [ -n "$md" ] || die "restore: array $UUID did not assemble"
    md_members | grep -q snvme && die "restore: $md assembled on snvme members?"

    if [ "$(mounted_src)" != "$md" ]; then
        if grep -qE "^[^#]*[[:space:]]$MOUNT[[:space:]]" /etc/fstab; then mount "$MOUNT" >>"$LOG" 2>&1
        else mount "$md" "$MOUNT" >>"$LOG" 2>&1; fi
    fi
    [ "$(mounted_src)" = "$md" ] || die "restore: mounting $md at $MOUNT failed"
    if [ -f "$STATE/docker_was_active" ]; then
        systemctl start docker >>"$LOG" 2>&1 && rm -f "$STATE/docker_was_active"
    fi
    rm -f "$STATE/chrdevs"
    echo restored > "$STATE/status"
    log "restore: done -- $md ($(md_members)) at $MOUNT on kernel nvme"
}

# ---------------------------------------------------------------------------
supervise() {  # runs detached; owns the daemon's lifetime
    exec 9>"$STATE/lock"; flock -n 9 || die "another supervisor is running"
    echo $$ > "$STATE/supervisor.pid"
    trap 'p=$(daemon_pid); alive "$p" && kill -TERM "$p"' TERM INT
    echo running > "$STATE/status"
    log "supervisor $$: starting daemon ($CONFIG)"
    (cd "$REPO" && TUTTI_VERBOSE=1 exec "$DAEMON" --config "$CONFIG") >>"$DLOG" 2>&1 &
    local pid=$!; echo "$pid" > "$STATE/daemon.pid"
    while alive "$pid"; do wait "$pid" 2>/dev/null; done
    log "supervisor: daemon $pid exited; restoring Linux md + nvme"
    restore || log "restore FAILED -- run: $0 $NAME restore"
    rm -f "$STATE/supervisor.pid"
}

# Top-level entries, minus the ACCEL<n> view directories the daemon publishes.
listing() { ls -A "$MOUNT" | grep -v '^ACCEL[0-9]*$' | sort | md5sum; }

healthy() {  # daemon serving + array mounted on snvme members + same listing
    local pid md; pid=$(daemon_pid); md=$(md_dev)
    alive "$pid" && ss -ltn 2>/dev/null | grep -q ":$PORT " && [ -n "$md" ] &&
        [ "$(md_members | wc -w)" = 4 ] && ! md_members | tr ' ' '\n' | grep -qv "^snvme\|^$" &&
        [ "$(mounted_src)" = "$md" ] &&
        [ "$(listing)" = "$(cat "$STATE/listing.md5")" ]
}

up() {
    alive "$(supervisor_pid)" && die "already up (supervisor $(supervisor_pid))"
    [ -x "$DAEMON" ] || die "$DAEMON missing"
    local md u; md=$(md_dev)
    [ -n "$md" ] && [ "$(mounted_src)" = "$md" ] || die "precondition: $MOUNT must be the array $UUID (run restore first)"
    md_members | grep -q snvme && die "precondition: array already on snvme members"
    if [ "$LEVEL" = protected ]; then
        array_clean "$md" || die "precondition: $md is not clean with all members (array_state=$(cat "/sys/block/$(basename "$md")/md/array_state" 2>/dev/null))"
        u=$(device_users)
        [ -z "$u" ] || die "precondition: $md or a member block device is open by $(describe $u)"
    fi
    listing > "$STATE/listing.md5"
    rm -f "$STATE/docker_was_active" "$STATE/chrdevs"
    if [ "$DOCKER" = 1 ] && systemctl is-active -q docker; then
        touch "$STATE/docker_was_active"; systemctl stop docker.socket docker >>"$LOG" 2>&1
    fi
    if [ "$LEVEL" = protected ]; then
        # Refuse, do not wait: whoever uses host data is stopped on purpose.
        u=$(mount_users)
        if [ -n "$u" ]; then
            if [ -f "$STATE/docker_was_active" ]; then
                systemctl start docker >>"$LOG" 2>&1 && rm -f "$STATE/docker_was_active"
            fi
            die "precondition: $MOUNT is in use by $(describe $u)"
        fi
        sync -f "$MOUNT"
    fi
    log "up: unmount $MOUNT, stop $md"
    umount_waiting || { restore; die "up: $MOUNT busy"; }
    mdadm --stop "$md" >>"$LOG" 2>&1 || { restore; die "up: mdadm --stop failed"; }
    echo starting > "$STATE/status"
    # The supervisor runs from a private copy: bash reads a script while it
    # runs, so editing this file must not change a live supervisor.
    install -m 755 "$0" "$STATE/tutti-md.sh"
    TUTTI_MD_REPO="$REPO" setsid "$STATE/tutti-md.sh" "$NAME" _supervise </dev/null >/dev/null 2>&1 &
    for _ in $(seq 1 120); do
        if healthy; then
            member_chrdevs > "$STATE/chrdevs"
            echo up > "$STATE/status"
            log "up: OK -- $(md_dev) ($(md_members)) at $MOUNT, daemon $(daemon_pid) :$PORT, controllers $(cat "$STATE/chrdevs")"
            return 0
        fi
        [ "$(cat "$STATE/status" 2>/dev/null)" = restored ] && die "up: daemon exited during startup; Linux md restored (see $DLOG)"
        sleep 1
    done
    log "up: health check failed after 120 s; rolling back"
    kill -TERM "$(supervisor_pid)" 2>/dev/null
    for _ in $(seq 1 600); do [ "$(cat "$STATE/status")" = restored ] && break; sleep 1; done
    die "up: failed; status=$(cat "$STATE/status")"
}

down() {
    local c u s
    c=$(chrdev_clients)
    [ -z "$c" ] || die "down: Tutti clients still drive $NAME's controllers: $(describe $c)-- stop them first"
    if [ "$LEVEL" = protected ]; then
        u=$(mount_users)
        [ -z "$u" ] || die "down: $MOUNT is in use by $(describe $u)-- stop them first (protected arrays are not waited out)"
    fi
    s=$(supervisor_pid)
    alive "$s" || { log "down: no supervisor; running restore directly"; restore; return; }
    kill -TERM "$s"
    for _ in $(seq 1 2000); do alive "$s" || break; sleep 1; done
    [ "$(cat "$STATE/status")" = restored ] || die "down: restore did not complete (see $LOG)"
    log "down: OK"
}

status() {
    echo "level=$LEVEL status=$(cat "$STATE/status" 2>/dev/null || echo unknown) supervisor=$(supervisor_pid) daemon=$(daemon_pid)"
    echo "array=$(md_dev) members=$(md_members) mount=$(mounted_src)"
    echo "controllers=$(member_chrdevs) clients=$(describe $(chrdev_clients))"
    for p in $PCIS; do echo -n "$p:$(driver_of "$p") "; done; echo
}

case "$CMD" in
up)         require_confirm up; up ;;
down)       require_confirm down; down ;;
restore)    require_confirm restore; restore ;;
status)     status ;;
_supervise) supervise ;;
*) usage; exit 2 ;;
esac
