{
  description = "NixOS edge AI system-userspace demo for Admiral / Jetson Orin";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  outputs = { self, nixpkgs }: let
    system = "aarch64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    os = nixpkgs.lib.nixosSystem {
      inherit system;
      specialArgs = { inherit self nixpkgs; };
      modules = [ ./nix/configuration.nix ];
    };
    closure = pkgs.closureInfo { rootPaths = [ os.config.system.build.toplevel ]; };
    root = pkgs.runCommand "admiral-nixos-root" {} ''
      mkdir -p $out/{etc,dev,proc,sys,run,tmp,var,usr/lib/aarch64-linux-gnu,lib,lib64,nix/var/nix/profiles}
      chmod 1777 $out/tmp
      ln -s ${os.config.system.build.toplevel}/init $out/init
      cp ${closure}/registration $out/nix-path-registration
      ln -s ${os.config.system.build.toplevel} $out/nix/var/nix/profiles/system
    '';
  in {
    nixosConfigurations.admiral-orin = os;
    nixosModules.admiral = import ./nix/admiral.nix;
    packages.${system} = {
      default = self.packages.${system}.image;
      image = pkgs.dockerTools.buildLayeredImage {
        name = "workload-nixos-demo";
        tag = "2026-09-23-nixos26.05-minicpm5-r3";
        architecture = "arm64";
        contents = [ root ];
        maxLayers = 32;
        config = {
          Entrypoint = [ "/init" ];
          Env = [ "container=oci" "PATH=/run/current-system/sw/bin:/nix/var/nix/profiles/default/bin" ];
          ExposedPorts = { "22/tcp" = {}; "8080/tcp" = {}; };
          StopSignal = "SIGRTMIN+3";
          Labels = {
            "org.opencontainers.image.title" = "Admiral NixOS Edge AI Demo";
            "org.opencontainers.image.source" = "https://github.com/admrlos/workload-nixos-demo";
            "org.opencontainers.image.version" = "2026-09-23-nixos26.05-minicpm5-r3";
            "co.admrl.workload" = "nixos-robotics";
          };
        };
      };
    };
  };
}
