#!/bin/sh

KERNEL=kernel7l
CONFIG=bcm2711_defconfig
JOBS=`grep processor /proc/cpuinfo|wc -l`
cdate=`date "+%Y-%m-%d"`

make LLVM=1 LLVM_IAS=1 ARCH=arm \
  CROSS_COMPILE=arm-linux-gnueabihf- $CONFIG
scripts/config -e LTO_CLANG_THIN
logsave build.log make -j$JOBS LLVM=1 LLVM_IAS=1 ARCH=arm \
  CROSS_COMPILE=arm-linux-gnueabihf- zImage modules dtbs
if [ -e arch/arm/boot/zImage ]; then
  mkdir -p out/boot/overlays
  make LLVM=1 ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- \
    INSTALL_MOD_PATH=out modules_install
  rm -rf out/lib/modules/*/build
  rm -rf out/lib/modules/*/source
  cp arch/arm/boot/dts/*.dtb out/boot/
  cp arch/arm/boot/dts/overlays/*.dtb* out/boot/overlays/
  cp arch/arm/boot/dts/overlays/README out/boot/overlays/
  cp arch/arm/boot/zImage out/boot/$KERNEL.img
  cd out
  zip -r pi4-kernel-$cdate.zip .
  cd ..
else
  echo "Build failed."
fi

