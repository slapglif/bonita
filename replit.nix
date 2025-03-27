{pkgs}: {
  deps = [
    pkgs.ffmpeg
    pkgs.playwright
    pkgs.glibcLocales
    pkgs.playwright-driver
    pkgs.gitFull
    pkgs.libxcrypt
    pkgs.bash
    pkgs.postgresql
    pkgs.openssl
    pkgs.nss
    pkgs.dbus
    pkgs.atk
    pkgs.at-spi2-atk
    pkgs.cups
    pkgs.expat
    pkgs.libXcomposite
    pkgs.libXdamage
    pkgs.libXfixes
    pkgs.mesa # For libgbm
    pkgs.libxcb
    pkgs.libxkbcommon
    pkgs.pango
    pkgs.cairo
    pkgs.systemd.lib # For libudev
    pkgs.alsa-lib
    pkgs.at-spi2-core
    
  ];
}
