def FormatBytesSize(num):
    """
    Given an integer value of bytes, convert to the most appropriate units
    ('bytes', 'kB', 'MB', 'GB', ...), and return a string containing the number
    of units and the unit label ('bytes', 'kB', 'MB', 'GB', ...)
    """

    base = 1024
    for unit in ['bytes', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']:
        if abs(num) < base:
            return "%3.2f %s" % (num, unit)
            break
        else:
            num /= base