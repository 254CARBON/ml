"""API subpackage for the model serving service.

Routes include synchronous prediction, batch jobs, and model discovery.
Designed to be thin layers over the ``ModelManager`` to keep business logic
out of transport code.
"""
