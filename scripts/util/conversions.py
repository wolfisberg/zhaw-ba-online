import mir_eval


def convert_cent_to_hz(c, fref=10.0):
    return fref * 2 ** (c / 1200.0)


def convert_hz_to_cent(f, fref=10.0):
    return mir_eval.melody.hz2cents(f, fref)


def convert_semitone_to_hz(c, fref=10.0):
    return convert_cent_to_hz(100 * c, fref)
