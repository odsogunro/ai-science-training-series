### YYYY-MM-DD

```
```

### 2022-09-27

MFA SSH on  NERSC,  https://bit.ly/3UeNcGj

- enter Password+OTP

```
- see 1password and authenticator
```

- download sshproxy.sh to local machine for 24 hour NERSC access 

```
$ scp osogunro@dtn01.nersc.gov:/global/cfs/cdirs/mfa/NERSC-MFA/sshproxy.sh .

$ ./sshproxy.sh -u osogunro

$ ls -l ~/.ssh/nersc*

$ ssh-keygen -L -f ~/.ssh/nersc-cert.pub | grep Valid
 
```

- PERFECT! login with the need for authentication

```
$ ssh -l osogunro -i ~/.ssh/nersc cori.nersc.gov
```

- help
```
$ ./sshproxy.sh -h
Usage: sshproxy.sh [-u <user>] [-o <filename>] [-s <scope>] [-c <account>] [-p] [-a] [-x <proxy-url>] [-U <server URL>] [-v] [-h]

         -u <user>      Specify remote (NERSC) username
                        (default: odsogunro)
         -o <filename>  Specify pathname for private key
                        (default: /home/odsogunro/.ssh/nersc)
         -s <scope>     Specify scope (default: 'default')
         -p             Get keys in PuTTY compatible (ppk) format
         -a             Add key to ssh-agent (with expiration)
         -c <account>   Specify a collaboration account (no default)
         -x <URL>       Use socks proxy to connect to sshproxy server.
                        (format: <protocol>://<host>[:port], see curl manpage
                        section on --proxy for details)
         -U <URL>       Specify alternate URL for sshproxy server
                        (generally only used for testing purposes)
         -v             Print version number and exit
         -h             Print this usage message and exit
```

- ...

```
$ ssh cori.nersc.gov

$ ssh perlmutter.nersc.gov 

$ ssh saul.nersc.gov 

$ ssh dtn.nersc.gov
```