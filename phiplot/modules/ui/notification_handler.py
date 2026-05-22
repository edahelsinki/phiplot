import logging

class NotificationHandler(logging.Handler):
    def __init__(self, notifications):
        super().__init__()
        self.notifications = notifications

    def emit(self, record):

        log_entry = self.format(record)
        level = record.levelname.lower()
        
        if level in ['error', 'critical']:
            msg_type = 'danger'
            duration = 5000
        elif level == 'warning':
            msg_type = 'warning'
            duration = 5000
        else:
            msg_type = 'info'
            duration = 3000

        self.notifications.send(log_entry, type=msg_type, duration=duration)