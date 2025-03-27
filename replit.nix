{pkgs}: {
  deps = [
    pkgs.glibcLocales
    pkgs.playwright-driver
    pkgs.gitFull
    pkgs.libxcrypt
    pkgs.bash
    pkgs.postgresql
    pkgs.openssl
  ];
}
