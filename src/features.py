def inputForm():
    pass


def buildContext(state):
    context = [0] * 12
    for note, notestate in enumerate(state):
        if notestate[0] == 1:
            pass
            # pitchclass = (note + lowerBound) % 12
            # context[pitchclass] += 1
    return context


def buildBeat(time):
    # returns a a length 4 vector unique to the position in the beat
    return [2 * x - 1 for x in [time % 2, (time // 2) % 2, (time // 4) % 2, (time // 8) % 2]]


def singleStatetoInput(state, time):
    beat = buildBeat(time)
    return [note for note in range(len(state))]


def stateMatrixtoInput(statematrix):
    # state is an nx2 matrix where the rows correspons to pitches
    inputform = [singleStatetoInput(state, time)
                 for time, state in enumerate(statematrix)]
    return inputform
