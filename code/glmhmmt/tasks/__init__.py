"""Drop-in task plugin package for editable installs of glmhmmt.

Add or remove ``*.py`` modules in this folder to change the task set shipped
with the repo. Each module should expose one or more ``TaskAdapter`` classes
registered via ``@_register([...])`` from ``glmhmmt.tasks``.
"""

GLMHMMT_TASK_PACKAGE = True
