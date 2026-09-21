{ config, lib, pkgs, modulesPath, ... }:
{
  imports = [ "${modulesPath}/virtualisation/docker-image.nix" ];
  # Admiral supplies the kernel, firmware, devices, network and GPU driver.
  boot.isContainer = true;
  networking.useDHCP = false;
  networking.useNetworkd = false;
  networking.useHostResolvConf = true;
  networking.resolvconf.enable = false;
  networking.firewall.enable = false;
  services.resolved.enable = false;
  services.udev.enable = false;
  systemd.services.systemd-networkd.enable = false;
  systemd.services.systemd-networkd-wait-online.enable = false;
  systemd.services.systemd-oomd.enable = false;
  systemd.oomd.enable = false;
  boot.enableContainers = false;
  # Never run NixOS's host-wide sysctl policy against Admiral's host /proc.
  systemd.services.systemd-sysctl.enable = false;
  documentation.enable = false;
  documentation.nixos.enable = false;
  nix.channel.enable = false;
  nix.settings.experimental-features = [ "nix-command" "flakes" ];
  nix.settings.trusted-users = [ "root" ];
  # Nix binaries do not search conventional /usr/lib paths. Explicitly expose
  # Admiral's payload to services and logins; glibc still comes from NixOS.
  environment.sessionVariables.LD_LIBRARY_PATH = "/run/admiral/nvidia/lib";
  systemd.globalEnvironment.LD_LIBRARY_PATH = "/run/admiral/nvidia/lib";
  users.groups.admiral-video.gid = 28;
  services.openssh = {
    enable = true;
    ports = [ 22 ];
    settings = {
      PasswordAuthentication = false;
      KbdInteractiveAuthentication = false;
      PermitRootLogin = "prohibit-password";
      AllowUsers = [ "root" "demo" ];
    };
  };
  system.stateVersion = "26.05";
}
