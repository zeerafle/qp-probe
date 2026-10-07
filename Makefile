PKGS := libavformat libavcodec libavutil

# Fail with something actionable instead of raw pkg-config noise. The binary is
# ABI-locked to the libavcodec it was built against, so it must be rebuilt on
# every machine -- do not copy a prebuilt qpprobe between distros.
ifeq ($(shell pkg-config --exists $(PKGS) && echo ok),)
$(warning )
$(warning ffmpeg development headers not found.)
$(warning )
$(warning   Debian/Ubuntu/Kaggle/Colab:)
$(warning     apt-get update && apt-get install -y libavformat-dev libavcodec-dev libavutil-dev pkg-config)
$(warning   Arch:    pacman -S ffmpeg pkgconf)
$(warning   macOS:   brew install ffmpeg pkg-config)
$(warning )
$(warning On Kaggle this needs notebook internet ON (Settings -> Internet).)
$(warning )
$(error cannot build without ffmpeg headers)
endif

CFLAGS := -O2 -Wall $(shell pkg-config --cflags $(PKGS))
LDLIBS := $(shell pkg-config --libs $(PKGS)) -lm

qpprobe: qpprobe.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
	@echo "built against libavcodec $$(pkg-config --modversion libavcodec)"

# Confirm this libavcodec actually exposes the side data we depend on.
# AV_FRAME_DATA_VIDEO_ENC_PARAMS landed in ffmpeg 4.3 (libavcodec 58.x).
check: qpprobe
	@pkg-config --atleast-version=58.91.100 libavcodec \
	  || { echo "libavcodec too old -- needs >= 4.3 for VIDEO_ENC_PARAMS"; exit 1; }
	@echo "libavcodec $$(pkg-config --modversion libavcodec) OK"

clean:
	rm -f qpprobe
	rm -rf out

# Ship source to Kaggle. `clean` first is not optional: the compiled qpprobe is
# ABI-locked to this machine's libavcodec, and shipping it is what made the
# notebook fail last time -- run.sh rebuilds from qpprobe.c on the target.
push: clean
	@grep -q KAGGLE_USERNAME dataset-metadata.json \
	  && { echo "set your username in dataset-metadata.json first"; exit 1; } || true
	kaggle datasets version -p . -m "$(M)" -r skip
	@echo
	@echo "Pushed. In the notebook the dataset is pinned to a version --"
	@echo "open the Input panel and update it, or restart the session."

# One-time, after filling in dataset-metadata.json.
create: clean
	kaggle datasets create -p . -r skip

.PHONY: clean check push create
