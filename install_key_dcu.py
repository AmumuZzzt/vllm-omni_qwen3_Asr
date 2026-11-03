#!/usr/bin/env python
"""Install the local public key into the remote root account (one-time, password auth).

The password comes from ADMIN_PW and is used only in memory: nothing is written
to disk, and nothing is echoed back.
"""

import os
import pathlib
import sys

import paramiko

HOST = "172.16.16.60"
USER = "root"
PUB = pathlib.Path.home() / ".ssh" / "id_dcu_node.pub"


def main() -> int:
    pw = os.environ.get("ADMIN_PW")
    if not pw:
        print("ADMIN_PW not set", file=sys.stderr)
        return 2
    if not PUB.exists():
        print(f"public key missing: {PUB}", file=sys.stderr)
        return 2

    pub = PUB.read_text(encoding="utf-8").strip()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, 22, USER, pw, timeout=25,
                       look_for_keys=False, allow_agent=False)
    except Exception as exc:
        print(f"CONNECT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    try:
        sftp = client.open_sftp()

        # ensure ~/.ssh with correct perms
        client.exec_command("mkdir -p /root/.ssh && chmod 700 /root/.ssh")[1].channel.recv_exit_status()

        try:
            with sftp.open("/root/.ssh/authorized_keys", "r") as f:
                existing = f.read().decode("utf-8", "replace")
        except IOError:
            existing = ""

        if pub in existing:
            print("公钥已存在，跳过写入")
        else:
            merged = (existing.rstrip("\n") + "\n" + pub + "\n") if existing.strip() else (pub + "\n")
            with sftp.open("/root/.ssh/authorized_keys", "w") as f:
                f.write(merged)
            print("公钥已追加")

        sftp.close()

        # fix perms + report
        stdin, stdout, stderr = client.exec_command(
            "chmod 600 /root/.ssh/authorized_keys; "
            "ls -l /root/.ssh/authorized_keys; "
            "echo -n 'key count: '; grep -c ssh-ed25519 /root/.ssh/authorized_keys",
            timeout=30,
        )
        print(stdout.read().decode("utf-8", "replace").strip())
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
