import os
import multiprocessing

bind = '0.0.0.0:80'
backlog = 2048

workers = multiprocessing.cpu_count()
worker_class = 'sync' or os.environ.get("WORKER_CLASS")
worker_connections = 1000
timeout = 30
keepalive = 2

reload = False
spew = False
daemon = False
raw_env = [
    "DB_HOST="+os.environ.get("DB_HOST", "127.0.0.1"),
    "DB_PORT="+os.environ.get("DB_PORT", "27017"),
    "DB_NAME="+os.environ.get("DB_NAME", "wi_regression"),
    "MODEL_DIR="+os.path.join(os.getcwd(), "static"),
]

if os.environ.get("DB_USER"):
    raw_env.append("DB_USER="+os.environ.get("DB_USER"))
    raw_env.append("DB_PASS="+os.environ.get("DB_PASS"))

pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

errorlog = '-'
loglevel = 'error'
accesslog = '/var/log/wi_regression.access.log'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

proc_name = None

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_fork(server, worker):
    pass

def pre_exec(server):
    server.log.info("Forked child, re-executing.")

def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

    ## get traceback info
    import threading, sys, traceback
    id2name = {th.ident: th.name for th in threading.enumerate()}
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name.get(threadId,""),
            threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename,
                lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    worker.log.debug("\n".join(code))

def worker_abort(worker):
    worker.log.info("Worker received SIGABRT signal")