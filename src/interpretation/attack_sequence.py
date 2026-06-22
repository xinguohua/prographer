"""
semanticthenTable. 

logthen: 
  - PROCESS_ROLE_MAP: process → feature
  - FILE_TYPE_MAP: fileextension/path → levelclass
  - NET_TYPE_MAP:  → 
  - EVENT_MAP: class → word
  - TYPE_MAP: nodeclass → 

then: 
  - INTENT_VERB_MAP: graphword → wordlist
  - INTENT_OBJECT_MAP: graphlevel → levelobjectclass
  - INTENT_SUBJECT_MAP: graphlevelprimary → levelprimary
"""
import re
from typing import List, Tuple, Optional

# ============================================================
# ============================================================

TYPE_MAP = {
    "SUBJECT_PROCESS": "process",
    "FILE_OBJECT_FILE": "file",
    "FILE_OBJECT_UNIX_SOCKET": "unix socket",
    "NetFlowObject": "network connection",
    "UnnamedPipeObject": "pipe",
    "SUBJECT_UNIT": "service unit",
    "FILE_OBJECT_DIR": "directory",
    "FILE_OBJECT_BLOCK": "block device",
    "FILE_OBJECT_CHAR": "character device",
    "RegistryKeyObject": "registry key",
    "SrcSinkObject": "source sink",
}

# ============================================================
# ============================================================

EVENT_MAP = {
    "EVENT_WRITE": "writes",
    "EVENT_READ": "reads",
    "EVENT_OPEN": "opens",
    "EVENT_CLOSE": "disables",
    "EVENT_EXECUTE": "executes",
    "EVENT_FORK": "executes process",
    "EVENT_EXIT": "exits",
    "EVENT_CONNECT": "sends network connection",
    "EVENT_SENDTO": "sends network connection",
    "EVENT_RECVFROM": "receives network connection",
    "EVENT_SENDMSG": "sends network connection",
    "EVENT_RECVMSG": "receives network connection",
    "EVENT_MODIFY_PROCESS": "writes process",
    "EVENT_CREATE_OBJECT": "creates",
    "EVENT_CHANGE_PRINCIPAL": "changes principal",
    "EVENT_LSEEK": "seeks in file",
    "EVENT_MODIFY_FILE_ATTRIBUTES": "writes configuration file",
    "EVENT_RENAME": "renames file",
    "EVENT_UNLINK": "deletes file",
    "EVENT_MMAP": "maps memory",
    "EVENT_MPROTECT": "changes memory protection",
    "EVENT_CLONE": "executes process",
    "EVENT_BIND": "sends network connection",
    "EVENT_ACCEPT": "receives network connection",
    "EVENT_LOGIN": "reads credential file",
    "EVENT_LOGOUT": "exits",
    "EVENT_LOADLIBRARY": "loads shared library",
    "EVENT_UPDATE": "writes",
    "EVENT_CHECK_FILE_ATTRIBUTES": "reads",
    "EVENT_READ_SOCKET_PARAMS": "reads network connection",
    "EVENT_WRITE_SOCKET_PARAMS": "writes network connection",
    "EVENT_DUP": "duplicates",
    "EVENT_OTHER": "executes",
}

# lowinfoprefix: translate_event returnbythiswordstart's resultwillfilter
LOW_INFO_PREFIXES = {"disables", "exits", "opens", "seeks", "maps memory",
                     "changes memory", "creates", "changes principal",
                     "duplicates"}

# ============================================================
# ============================================================

PROCESS_ROLE_MAP = {
    # command shell (data: ATT&CK T1059.003/004) 
    "bash": "command shell",
    "sh": "command shell",
    "zsh": "command shell",
    "csh": "command shell",
    "tcsh": "command shell",
    "dash": "command shell",
    "fish": "command shell",
    "cmd": "command shell",
    "cmd.exe": "command shell",
    # scripting interpreter (data: ATT&CK T1059.001/005/006, ~103tuple) 
    "python": "scripting interpreter",
    "python2": "scripting interpreter",
    "python3": "scripting interpreter",
    "perl": "scripting interpreter",
    "ruby": "scripting interpreter",
    "node": "scripting interpreter",
    "php": "scripting interpreter",
    "powershell": "scripting interpreter",
    "pwsh": "scripting interpreter",
    "powershell.exe": "scripting interpreter",
    # remote access service (data: ATT&CK T1021, ~53tuple) 
    "sshd": "remote access service",
    "ssh": "remote access service",
    "telnetd": "remote access service",
    # proxy executor / LOLBins (data: ATT&CK T1218, ~153tuple) 
    "rundll32": "proxy executor",
    "rundll32.exe": "proxy executor",
    "regsvr32": "proxy executor",
    "regsvr32.exe": "proxy executor",
    "cmstp": "proxy executor",
    "cmstp.exe": "proxy executor",
    "msiexec": "proxy executor",
    "msiexec.exe": "proxy executor",
    "installutil": "proxy executor",
    "installutil.exe": "proxy executor",
    "mavinject": "proxy executor",
    "mavinject.exe": "proxy executor",
    "odbcconf": "proxy executor",
    "odbcconf.exe": "proxy executor",
    "wmic": "proxy executor",
    "wmic.exe": "proxy executor",
    "mshta": "proxy executor",
    "mshta.exe": "proxy executor",
    "certutil": "proxy executor",
    "certutil.exe": "proxy executor",
    # itsremainderprocessnotmapping, preserve (Sentence-BERT connectmatch) 
}

# ============================================================
# ============================================================

FILE_EXT_MAP = {
    # shared library
    ".so": "shared library",
    ".dll": "shared library",
    ".dylib": "shared library",
    # configuration file
    ".conf": "configuration file",
    ".cfg": "configuration file",
    ".ini": "configuration file",
    ".yaml": "configuration file",
    ".yml": "configuration file",
    # log file
    ".log": "log file",
    ".evtx": "log file",
    # credential / key file
    ".pem": "authentication key file",
    ".key": "authentication key file",
    ".crt": "authentication key file",
    ".cer": "authentication key file",
    ".pgp": "authentication key file",
    ".gpg": "authentication key file",
    # executable
    ".exe": "executable",
    ".elf": "executable",
    # script → executable
    ".sh": "executable",
    ".py": "executable",
    ".pl": "executable",
    ".rb": "executable",
    ".js": "executable",
    ".vbs": "executable",
    ".ps1": "executable",
    ".bat": "executable",
    ".cmd": "executable",
    # data → file
    ".db": "file",
    ".sqlite": "file",
    ".json": "file",
    ".xml": "file",
    ".csv": "file",
    # archive → file
    ".zip": "file",
    ".tar": "file",
    ".gz": "file",
    ".bz2": "file",
    ".7z": "file",
    ".rar": "file",
}

# ============================================================
# ============================================================

FILE_PATH_MAP = [
    # credential files (specificpathpriority first match)
    ("/etc/shadow", "credential file"),
    ("/etc/passwd", "credential file"),
    ("/etc/master.passwd", "credential file"),
    # authorized keys
    ("authorized_keys", "authentication key file"),
    (".ssh/", "configuration file"),
    # scheduled task → configuration file (17 kindclassinwithout's  scheduled task class) 
    ("crontab", "configuration file"),
    ("/etc/cron", "configuration file"),
    ("/etc/init.d/", "configuration file"),
    ("/etc/systemd/", "configuration file"),
    # proc filesystem → process
    ("/proc/", "process"),
    # log
    ("/var/log/", "log file"),
    # general paths → mappingis17 kindclassin's class
    ("/tmp/", "file"),
    ("/etc/", "configuration file"),
    ("/dev/", "file"),
    ("/bin/", "executable"),
    ("/sbin/", "executable"),
    ("/usr/bin/", "executable"),
    ("/home/", "file"),
    ("/root/", "file"),
]

# ============================================================
# ============================================================

PORT_MAP = {
    "80": "network connection",
    "443": "network connection",
    "22": "network connection",
    "53": "network connection",
    "25": "email",
    "587": "email",
    "110": "network connection",
    "143": "network connection",
    "21": "network connection",
    "23": "network connection",
    "3306": "network connection",
    "5432": "network connection",
    "6379": "network connection",
    "3389": "network connection",
    "445": "network connection",
    "139": "network connection",
    "8080": "network connection",
    "8443": "network connection",
}

# ============================================================
# ============================================================

INTENT_SUBJECT_MAP = {
    "adversaries": "process",
    "adversary": "process",
    "threat actors": "process",
    "threat actor": "process",
    "attackers": "process",
    "attacker": "process",
    "victims": "process",
    "victim": "process",
    "legitimate users": "user",
    "legitimate user": "user",
    "an adversary": "process",
}

# ============================================================
# ============================================================

INTENT_VERB_MAP = {
    "inject": ["writes", "writes", "reads"],
    "exfiltrate": ["reads", "sends"],
    "persist": ["writes"],
    "establish": ["writes"],
    "dump": ["reads", "reads", "writes"],
    "escalate": ["reads", "executes"],
    "elevate": ["reads", "executes"],
    "steal": ["reads", "sends"],
    "hijack": ["reads", "writes"],
    "enumerate": ["reads"],
    "discover": ["reads"],
    "collect": ["reads"],
    "encrypt": ["reads", "writes"],
    "obfuscate": ["reads", "writes"],
    "masquerade": ["writes", "renames"],
    "impersonate": ["reads", "executes"],
    "capture": ["reads"],
    "harvest": ["reads"],
    "compromise": ["reads", "writes", "executes"],
    "exploit": ["reads", "executes"],
    "abuse": ["executes"],
    "leverage": ["executes"],
    "deploy": ["writes", "executes"],
    "deliver": ["writes", "sends"],
    "stage": ["writes"],
    "scan": ["sends", "receives"],
    "sniff": ["reads"],
    "intercept": ["reads"],
    "tamper": ["writes"],
    "modify": ["writes"],
    "create": ["writes"],
    "delete": ["deletes"],
    "disable": ["writes"],
    "clear": ["deletes", "writes"],
}

# ============================================================
# ============================================================

INTENT_OBJECT_MAP = {
    "code": "shared library",
    "malicious code": "shared library",
    "arbitrary code": "shared library",
    "processes": "process memory",
    "process memory": "process memory",
    "process": "process memory",
    "dll": "shared library",
    "shared library": "shared library",
    "shared module": "shared library",
    "executable": "executable",
    "binary": "executable",
    "payload": "executable",
    "credentials": "credential file",
    "passwords": "credential file",
    "password": "credential file",
    "credential": "credential file",
    "hashes": "credential file",
    "tokens": "authentication token",
    "token": "authentication token",
    "keys": "authentication key file",
    "key": "authentication key file",
    "kerberos ticket": "authentication token",
    "certificates": "certificate file",
    "data": "file",
    "collected data": "file",
    "files": "file",
    "file": "file",
    "documents": "file",
    "registry": "configuration file",
    "registry key": "configuration file",
    "configuration": "configuration file",
    "scheduled task": "scheduled task configuration",
    "cron job": "scheduled task configuration",
    "startup entry": "scheduled task configuration",
    "service": "service configuration",
    "log": "log file",
    "logs": "log file",
    "event logs": "log file",
    "network": "network data",
    "network data": "network data",
    "traffic": "network data",
    "c2 channel": "network connection",
    "command and control": "network connection",
    "remote server": "remote connection",
    "remote service": "remote connection",
    "remote services": "remote connection",
    "email": "email",
    "phishing": "email",
}

# ============================================================
# ============================================================

def get_process_role(process_name: str) -> str:
    """process → feature. hitthen. """
    name = process_name.strip().lower()
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return PROCESS_ROLE_MAP.get(name, name)


def get_file_type(filepath: str) -> str:
    """filepath → levelclass. extensionpriority first inpathprefix. """
    fp = filepath.strip()

    # first matchextension (priority first , is .so/.dll  vs  /tmp/ morehasinfo) 
    for ext, desc in FILE_EXT_MAP.items():
        if fp.endswith(ext):
            return desc

    for prefix, desc in FILE_PATH_MAP:
        if prefix in fp:
            return desc

    return ""


def get_port_protocol(port: str) -> str:
    """ → . """
    return PORT_MAP.get(port.strip(), "")


def is_internal_ip(ip: str) -> bool:
    """break IP isisinsidely (RFC 1918) . """
    ip = ip.strip()
    return (ip.startswith("10.") or
            ip.startswith("192.168.") or
            ip.startswith("172.16.") or ip.startswith("172.17.") or
            ip.startswith("172.18.") or ip.startswith("172.19.") or
            ip.startswith("172.2") or ip.startswith("172.30.") or
            ip.startswith("172.31.") or
            ip.startswith("127.") or
            ip == "localhost")


def translate_event(event_str: str) -> str:
    """log: willuseis. 

    input: 'EVENT_WRITE /tmp/memhelp.so'
    output: 'writes shared library'

    outputis "verb object_type", and3tuple's  verb+object partalign. 
    """
    event_str = event_str.strip()
    if not event_str:
        return ""

    parts = event_str.split(None, 1)
    event_type = parts[0]
    obj = parts[1] if len(parts) > 1 else ""

    action = EVENT_MAP.get(event_type, "")
    if not action:
        return ""

    # ifalreadyviacontainsinteger's  "verb object_type" (like "sends network connection") , connectreturn
    if " " in action:
        return action

    if not obj:
        return action

    file_type = get_file_type(obj)
    if file_type:
        return f"{action} {obj} {file_type}"

    return f"{action} {obj}"


# ============================================================
# ============================================================

def map_intent_subject(subject: str) -> str:
    """: graphlevelprimary → levelprimary. """
    s = subject.strip().lower()
    return INTENT_SUBJECT_MAP.get(s, "process")


def map_intent_verb(verb: str) -> Optional[List[str]]:
    """: graphword → wordlist. notinTableinreturn None (drop3tuple) . """
    v = verb.strip().lower()
    if v in INTENT_VERB_MAP:
        return INTENT_VERB_MAP[v]
    # wordmatch (like injecting → inject) 
    for intent_v, ops in INTENT_VERB_MAP.items():
        if v.startswith(intent_v) or intent_v.startswith(v):
            return ops
    return None


def map_intent_object(obj: str) -> str:
    """: graphlevel → levelobjectclass. """
    o = obj.strip().lower()
    for modifier in ["malicious", "stolen", "legitimate", "arbitrary",
                     "suspicious", "unauthorized", "compromised",
                     "sensitive", "valid", "additional"]:
        o = o.replace(modifier, "").strip()

    if o in INTENT_OBJECT_MAP:
        return INTENT_OBJECT_MAP[o]
    for key, val in INTENT_OBJECT_MAP.items():
        if key in o:
            return val
    return o
