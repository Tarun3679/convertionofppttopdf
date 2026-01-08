cat > /tmp/container_diagnostics.sh << 'SCRIPT_EOF'
#!/bin/bash

echo "=========================================="
echo "Container Environment Diagnostics"
echo "=========================================="
echo ""

echo "=== 1. OS Information ==="
cat /etc/redhat-release 2>&1 || echo "redhat-release not found"
uname -r
python3 --version 2>&1 || echo "python3 not found"
echo ""

echo "=== 2. User & Permissions ==="
id
echo ""

echo "=== 3. Cgroup Version Detection ==="
echo "Mount info:"
mount | grep cgroup
echo ""
echo "Filesystem type:"
stat -fc %T /sys/fs/cgroup/ 2>&1
echo ""

echo "=== 4. Cgroup Path ==="
cat /proc/self/cgroup
echo ""

echo "=== 5. Cgroup Memory Files (v1) ==="
ls -lh /sys/fs/cgroup/memory/memory.* 2>&1 | head -10
echo ""

echo "=== 6. Cgroup Memory Files (v2) ==="
ls -lh /sys/fs/cgroup/memory.* 2>&1
echo ""

echo "=== 7. Memory Readings ==="
echo -n "v1 usage: "
cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>&1 | awk '{print $1/1024/1024 " MB"}' 2>&1
echo -n "v1 limit: "
cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>&1 | awk '{print $1/1024/1024 " MB"}' 2>&1
echo -n "v2 usage: "
cat /sys/fs/cgroup/memory.current 2>&1 | awk '{print $1/1024/1024 " MB"}' 2>&1
echo -n "v2 limit: "
cat /sys/fs/cgroup/memory.max 2>&1
echo ""

echo "=== 8. Process Info ==="
echo "Processes visible in /proc:"
ls /proc | grep -E '^[0-9]+$' | wc -l
echo "Can read /proc/1/status:"
head -5 /proc/1/status 2>&1
echo "Self memory:"
cat /proc/self/status | grep -E 'VmRSS|VmSize'
echo ""

echo "=== 9. Environment Variables ==="
env | grep -E 'KUBERNETES|OPENSHIFT|PODMAN|DOCKER|CONTAINER' | sort
echo ""

echo "=== 10. Children Tracking ==="
echo "Current PID: $$"
echo -n "Children file exists: "
[ -f /proc/$$/task/$$/children ] && echo "YES" || echo "NO"
cat /proc/$$/task/$$/children 2>&1 || echo "Cannot read children file"
echo ""

echo "=========================================="
echo "Diagnostics Complete"
echo "=========================================="
SCRIPT_EOF

chmod +x /tmp/container_diagnostics.sh
/tmp/container_diagnostics.sh
