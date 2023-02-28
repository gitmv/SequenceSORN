import os

os.system('start /wait cmd /c ssh root@192.168.1.10 -p 9222 screen -m -r 62 "tail -f /var/log/messages"')#"screen -list; exec /bin/sh"