import midi
import numpy as np

lowerBound = 16  # midi pitch corresponding to just above 20 Hz
upperBound = 135  # midi pitch corresponding to just below 20 kHz


def midiToStateMatrix(midifile):
    '''
    INPUT: str
    OUTPUT: numpy array

    Take in the path to a midi file and output it as a state matrix
    '''

    # pattern is the container of the entire midi file, and contains
    # the Tempo and Resolution of the music and a list of tracks
    pattern = midi.read_midifile(midifile)

    # this tracks time until next known event in each track
    timeleft = [track[0].tick for track in pattern]

    positions = [0 for track in pattern]  # i don't know yet

    statematrix = []

    # the range of human hearing in integer midi pitches
    span = upperBound - lowerBound
    time = 0

    # the state of each note is a vector of length 2. n[0] is on, n[1]
    # means it's a new note, so a note of [1,1] is a new note that
    # stops the previous note at that pitch
    state = [[0, 0] for x in range(span)]
    statematrix.append(state)

    while True:  # enter file reading loop
        # watch within a 16th note range for time reaching a new 32nd
        # note to make new state, this means the smallest note
        # possible is a 32nd note
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            oldstate = state
            # the new state is assumed to just be the old state with
            # all notes held
            state = [[oldstate[x][0], 0] for x in range(span)]
            statematrix.append(state)

        # this for loop has the effect of compressing all the tracks
        # into a single statematrix
        for i in range(len(timeleft)):  # for each track
            while timeleft[i] == 0:
                track = pattern[i]  # the track we're looking at
                # initialize position in the track
                position = positions[i]

                event = track[position]  # first event
                if isinstance(event, midi.NoteEvent):
                    if (event.pitch < lowerBound) or (event.pitch >= upperBound):
                        pass
                        # print "Note {} at time {} out of bounds
                        # (ignoring)".format(event.pitch, time)
                    else:
                        if isinstance(event, midi.NoteOffEvent) or event.velocity == 0:
                            state[event.pitch - lowerBound] = [0, 0]
                        else:  # if the event is not a note stop, make new note
                            state[event.pitch - lowerBound] = [1, 1]
                elif isinstance(event, midi.TimeSignatureEvent):
                    if event.numerator not in (2, 4):
                        # We don't want to worry about non-4 time
                        # signatures. Bail early!
                        print "Found time signature event {}. Bailing!".format(event)
                        return statematrix

                try:
                    # set time to next known event
                    timeleft[i] = track[position + 1].tick
                    positions[i] += 1  # iterate positions forward
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        # if there are no events left in any track, break the loop
        if all(t is None for t in timeleft):
            break

        time += 1

    return statematrix


def noteStateMatrixToMidi(statematrix, filename="example"):
    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    span = upperBound - lowerBound
    tickscale = 55

    lastcmdtime = 0
    previousstate = [[0, 0] for x in range(span)]
    # loop through every state in the statematrix
    for time, state in enumerate(statematrix + [previousstate[:]]):
        offNotes, onNotes = [], []
        # for each note determine if turning it off or on or both
        for i in range(span):
            n = state[i]
            p = previousstate[i]
            if p[0] == 1:  # index 0 is on/off
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:  # index 1 id new/old
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(
                tick=(time - lastcmdtime) * tickscale, pitch=note + lowerBound))
            lastcmdtime = time  # event tick is relative to previous event
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(
                time - lastcmdtime) * tickscale, velocity=40, pitch=note + lowerBound))
            lastcmdtime = time  # event tick is relative to previous event

        previousstate = state  # update previousstate

    # midi files require an end f track event to be run by most
    # programs
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(filename), pattern)
