{ config, lib, pkgs, self, nixpkgs, ... }:
let
  cudaPkgs = import nixpkgs {
    system = "aarch64-linux";
    config = { allowUnfree = true; cudaCapabilities = [ "8.7" ]; };
  };
  llama = cudaPkgs.llama-cpp.override {
    cudaSupport = true;
    cudaPackages = cudaPkgs.cudaPackages_12_6.overrideScope (_: _: {
      # Admiral owns libcuda; never ship a replacement compatibility driver.
      cuda_compat = null;
    });
  };
  model = pkgs.fetchurl {
    url = "https://huggingface.co/openbmb/MiniCPM5-2B-GGUF/resolve/2079a22f3beaa4e306449978533478fe0522f4b3/MiniCPM5-2B-Q4_K_M.gguf";
    sha256 = "ec2d5801640099e97d8d7e8003ad4d81f336e757811f03a26173dddf386602fd";
  };
  app = pkgs.stdenvNoCC.mkDerivation {
    pname = "admiral-robotics-demo";
    version = "0.2.0";
    src = ../demo;
    nativeBuildInputs = [ pkgs.makeWrapper ];
    installPhase = ''
      mkdir -p $out/share/admiral-demo $out/bin
      cp -r . $out/share/admiral-demo/
      makeWrapper ${pkgs.python3}/bin/python3 $out/bin/demo-server \
        --add-flags "$out/share/admiral-demo/server.py" \
        --prefix LD_LIBRARY_PATH : /run/admiral/nvidia/lib
      makeWrapper ${pkgs.python3}/bin/python3 $out/bin/demo-gpu \
        --add-flags "$out/share/admiral-demo/gpu.py" \
        --prefix LD_LIBRARY_PATH : /run/admiral/nvidia/lib
      makeWrapper ${pkgs.python3}/bin/python3 $out/bin/demo-chat \
        --add-flags "$out/share/admiral-demo/chat.py" \
        --prefix LD_LIBRARY_PATH : /run/admiral/nvidia/lib
    '';
  };
  status = pkgs.writeShellScriptBin "demo-status" ''
    echo '=== NixOS on Admiral ==='
    cat /etc/os-release
    echo; uname -a; id
    echo; ${pkgs.nix}/bin/nix --version
    echo; ${pkgs.systemd}/bin/systemctl --no-pager status robotics-demo.service || true
    echo; ${pkgs.curl}/bin/curl --fail --silent http://127.0.0.1:8080/api/status | ${pkgs.jq}/bin/jq
  '';
  source = pkgs.runCommand "admiral-demo-source" {} ''
    mkdir -p $out
    cp ${../flake.nix} $out/flake.nix
    cp ${../flake.lock} $out/flake.lock
    cp -r ${../nix} $out/nix
    cp -r ${../demo} $out/demo
    cp -r ${../ssh} $out/ssh
  '';
in {
  imports = [ ./admiral.nix ];
  networking.hostName = "admiral-nixos";
  nix.registry.nixpkgs.flake = nixpkgs;
  nix.nixPath = [ "nixpkgs=${nixpkgs}" ];
  environment.systemPackages = with pkgs; [
    app status bashInteractive curl jq gitMinimal nano htop procps iproute2
    iputils util-linux strace binutils file nix hello
  ];
  users.users.root.openssh.authorizedKeys.keyFiles = [ ../ssh/authorized_keys ];
  users.users.demo = {
    isNormalUser = true;
    uid = 1000;
    extraGroups = [ "admiral-video" ];
    openssh.authorizedKeys.keyFiles = [ ../ssh/authorized_keys ];
  };
  environment.etc."admiral-demo/version".text = "2026-09-23-nixos26.05-minicpm5-r2\n";
  environment.etc."admiral-demo/flake.lock".source = ../flake.lock;
  environment.etc."admiral-demo/source".source = source;
  environment.etc."motd".text = ''

    ADMIRAL / NIXOS EDGE AI LAB
    Dashboard : http://DEVICE_IP:8080
    Inspect   : demo-status
    Chat CLI  : demo-chat "Your question"
    GPU test  : timeout 30 demo-gpu
    Services  : systemctl status robotics-demo
    Logs      : journalctl -u robotics-demo -f
    Nix       : nix --version; nix registry list

    MiniCPM5-2B chat running on Jetson Orin via llama.cpp / CUDA.
  '';
  systemd.services.minicpm = {
    description = "MiniCPM5-2B llama.cpp CUDA inference (all layers on Orin GPU)";
    wantedBy = [ "multi-user.target" ];
    after = [ "network.target" ];
    environment = {
      LD_LIBRARY_PATH = "/run/admiral/nvidia/lib";
      PYTHONUNBUFFERED = "1";
    };
    serviceConfig = {
      ExecStart = "${pkgs.python3}/bin/python3 ${app}/share/admiral-demo/inference_service.py ${llama}/bin/llama-server --model ${model} --alias openbmb/MiniCPM5-2B --host 127.0.0.1 --port 8081 --device CUDA0 --n-gpu-layers 999 --fit off --ctx-size 8192 --parallel 1 --jinja --reasoning off --temp 1.0 --top-p 0.95 --min-p 0.0";
      User = "demo";
      Group = "users";
      SupplementaryGroups = [ "admiral-video" ];
      RuntimeDirectory = "minicpm";
      Restart = "on-failure";
      RestartSec = 10;
      NoNewPrivileges = true;
      ProtectSystem = "strict";
      ProtectHome = true;
      PrivateTmp = true;
    };
  };
  systemd.services.robotics-demo = {
    description = "Admiral NixOS edge LLM chat dashboard and checked CUDA probe";
    wantedBy = [ "multi-user.target" ];
    after = [ "network.target" ];
    environment = {
      LD_LIBRARY_PATH = "/run/admiral/nvidia/lib";
      PYTHONUNBUFFERED = "1";
    };
    serviceConfig = {
      ExecStart = "${app}/bin/demo-server";
      User = "demo";
      Group = "users";
      SupplementaryGroups = [ "admiral-video" ];
      Restart = "on-failure";
      RestartSec = 2;
      StateDirectory = "admiral-demo";
      NoNewPrivileges = true;
      ProtectSystem = "strict";
      ProtectHome = true;
      PrivateTmp = true;
      # Devices stay visible: Admiral owns their lifecycle and access policy.
    };
  };
}
