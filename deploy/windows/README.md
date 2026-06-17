# iobot dashboard — Windows launchers

Read-only monitor for the iobot intraday options bot. The bot runs on the EC2 VM
(systemd); these scripts only open/close a local SSH tunnel so you can view the
dashboard in your browser. Closing the tunnel never stops the bot.

## One-time setup
1. Copy both `.bat` files to your Desktop (or a folder of your choice).
2. Make sure your SSH key is at `%USERPROFILE%\Desktop\Options-bot.pem`
   (e.g. `C:\Users\Shaq\Desktop\Options-bot.pem`). If it's elsewhere, edit the
   `set KEY=` line in `Open iobot Dashboard.bat`.

## Use
- **Open iobot Dashboard.bat** — opens the tunnel and your browser at
  http://localhost:8501. Leave the `iobot-tunnel` window open while viewing.
- **Close iobot Dashboard.bat** — closes the tunnel only.

Server details: `ubuntu@3.87.108.0`, dashboard bound to `127.0.0.1:8501` on the VM
(never exposed publicly; AWS security group allows SSH/22 only).
