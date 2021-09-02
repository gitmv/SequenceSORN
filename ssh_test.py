import paramiko
import time
import random
from scp import SCPClient
import os

def get_response(out, err):
    result = []
    for line in out.read().splitlines():
        result.append(line.decode("utf-8"))
    for line in err.read().splitlines():
        result.append(line.decode("utf-8"))
    return result

def get_ssh_connection(host, user, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, password=password)
    return ssh

#for i in range(10000):
ssh = get_ssh_connection('192.168.1.14', 'marius', None)#poppy.fias.uni-frankfurt.de
#    print(i)

scp = SCPClient(ssh.get_transport())
scp.put('nest_embedding2.py', 'nest_embedding2.py')
#scp.get('nest_embedding2.py', 'nest_embedding2.py')
scp.close()

#os.remove("test.txt")

#scp = SCPClient(ssh.get_transport())
#scp.get('test.txt', 'test.txt')
#scp.close()

#cmd = 'rm test.txt'
#cmd='ls'
#ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd)
#print(i, get_response(ssh_stdout, ssh_stderr))

#ssh.close()

#time.sleep(random.random()/10)
