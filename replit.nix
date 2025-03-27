{pkgs}: {
  deps = [
    pkgs.python311Packages.loguru
    pkgs.python311Packages.rich
    pkgs.python311Packages.pytest-playwright
    pkgs.playwright-test
    pkgs.python311Packages.playwright
    pkgs.python311Packages.playwright-stealth
    pkgs.python311Packages.playwrightcapture
    pkgs.ffmpeg
    pkgs.playwright
    pkgs.glibcLocales
    pkgs.playwright-driver
    pkgs.gitFull
    pkgs.libxcrypt
    pkgs.bash
    pkgs.postgresql
    pkgs.openssl
    
    
  ];
}
