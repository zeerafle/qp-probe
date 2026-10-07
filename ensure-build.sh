# Sourced by run.sh and sweep.sh.
#
# The binary is ABI-locked to its build machine's libavcodec. A qpprobe copied
# in from elsewhere (or one whose +x bit survived a dataset upload) will either
# fail to exec or crash, so smoke-test it and rebuild rather than trusting it.
if ! ./qpprobe 2>/dev/null; [ $? -ne 2 ]; then
  echo "building qpprobe for this machine..."
  make clean >/dev/null 2>&1
  make || { echo "build failed -- see the Makefile message above."; exit 1; }
fi
