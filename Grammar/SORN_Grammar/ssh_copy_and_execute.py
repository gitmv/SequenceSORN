from PymoNNto import *
from PymoNNto.Exploration.Evolution.SSH_Functions import *
import random

target = 'ssh marius@hey3kmuagjunsk2b.myfritz.net' #'ssh vieth@poppy.fias.uni-frankfurt.de'
file = 'Grammar/SORN_Grammar/WTA_SORN.py'

user, host, password = split_ssh_user_host_password_string()
id = str(random.randint(1, 10000))
clone_project(id)
transfer_project(id, user, host, password)



ssh = get_ssh_connection(host, user, password)

command = 'cd ' + id + '; '
# command = 'nano .bashrc'
command += 'screen -dmS ' + id + ' sh; screen -S ' + id + ' -X stuff "' + python_cmd + ' execute_evolution.py \r\n"'

ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
response = get_response(ssh_stdout, ssh_stderr)
print(response)

ssh.close()





