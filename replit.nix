{pkgs}: {
  deps = [
    pkgs.python311Packages.gunicorn
    pkgs.python312Packages.gunicorn
    pkgs.python312Packages.langchain-text-splitters
    pkgs.python311Packages.langchain-text-splitters
    pkgs.python311Packages.langchain-community
    pkgs.python312Packages.langchain-core
    pkgs.python312Packages.langchain-community
    pkgs.python311Packages.langchain-core
    pkgs.python311Packages.langchain
    pkgs.python312Packages.langchain
    pkgs.python312Packages.unstructured
    pkgs.python311Packages.unstructured
    pkgs.python312Packages.jsonnet
    pkgs.opencv
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
