"""Project task package used by glmhmmt.

Add or remove ``*.py`` modules in this folder to change the task set used by a
repo checkout or any working directory configured through ``GLMHMMT_TASK_PATHS``
or ``[plugins].task_paths``. Each module should expose one or more
``TaskAdapter`` classes registered via ``@_register([...])`` from
``glmhmmt.tasks``.
"""

GLMHMMT_TASK_PACKAGE = True
