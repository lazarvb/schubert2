from music21 import *
from fractions import Fraction
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import json

'''
This particularly high number of functions and lines of code (more than 400) is mainly due to the way
the scores are written in "voices". Voices correspond to melodic lines that occur simultaneously within
a single measure and sometimes within the same hand. Thus, a single measure can contain several independent
voices, which artificially multiplies the number of events to be processed in the score. To simplify the
analysis and reduce complexity, I chose in my project to merge all these voices into a single musical line
per hand.

1. Extraction of events

The first function, extract_events_from_score(), extracts all the musical elements from a score and sorts
them according to their order of appearance within each measure. Each note, chord, or rest is then represented
in a structured form (hand, temporal position, type of event). This step also includes handling of time
signatures, key signatures, and transposition of the score to C major (or A minor) to make the processing consistent.

2. Merging and simplification of events

The merge_events_simple() function relies on process_buffer() to group together events that occur at the same
moment. For example, several simultaneous notes are merged into a single chord. Thus, each musical instant is
no longer described by several independent voices, but by a single object representing the global chord at that
moment. At the end of this step, each event is represented by a tuple of the form (hand, note, duration), which
corresponds to the natural structure of .mxl files.

3. Conversion into tokens

The events_to_tokens2() function transforms this sequence of tuples into a sequence of tokens. This step aims
to reduce the size of the dictionary used by machine learning models by eliminating redundancies. The hand name
(right hand or left hand) is indicated only at the beginning of each measure or for the left hand in the middle
of the measure, and the following events simply list the note names and their durations. This compact format is
better suited for sequential processing while preserving the essential musical structure.

4. Reverse decoding

The tokens_to_events2() and decode_to_score() functions perform the reverse process. They make it possible to
reconstruct a readable score in .mxl format from a sequence of tokens, recreating the structure of measures,
time signatures, and musical events (notes, chords, rests).

'''

def extract_events_from_score(filepath, n_hand):
    score = converter.parse(filepath)
    
    armures = []
    for part in score.parts:
        for ks in part.recurse().getElementsByClass(key.KeySignature):
            armures.append(ks)
    tonalite = armures[0].asKey()
    
    i = interval.Interval(tonalite.tonic, pitch.Pitch('C'))
    score = score.transpose(i)
    # This part was used for the transposition to C major/A minor

    events = []

    if len(score.parts) != n_hand:
        raise ValueError(f"{len(score.parts)} parts")

    for i, part in enumerate(score.parts):
        part.partName = "main_droite" if i == 0 else "main_gauche"

    # Group the measurements by number
    measures_by_number = {}
    for part in score.parts:
        part_name = part.partName
        for m in part.getElementsByClass(stream.Measure):
            if m.number not in measures_by_number:
                measures_by_number[m.number] = []
            measures_by_number[m.number].append((part_name, m))

    # Follow the previous timestamp
    current_time_signature = None

    for measure_number in sorted(measures_by_number.keys()):
        events.append(("barline", f"measure_{measure_number}"))

        first_part_measure = measures_by_number[measure_number][0][1]
        ts = first_part_measure.timeSignature
        if ts is not None:
            ts_str = f"{ts.numerator}/{ts.denominator}"
            if current_time_signature is None or ts_str != current_time_signature:
                events.append(("time_signature", ts_str))
                current_time_signature = ts_str

        # Temporary list to store the measurement events
        measure_events = []

        for part_name, measure in measures_by_number[measure_number]:
            voices = measure.getElementsByClass(stream.Voice)
            if not voices:
                voices = [measure]

            for voice in voices:
                for el in voice.notesAndRests:
                    if el.duration.isGrace:
                        continue

                    offset = float(el.offset)
                    if isinstance(el, (note.Note, chord.Chord, note.Rest)):
                        if isinstance(el, note.Note):
                            pitches = sorted({el.pitch.midi})
                            measure_events.append((part_name, offset, f"note_{'-'.join(map(str, pitches))}"))
                        elif isinstance(el, chord.Chord):
                            if len(el.notes) > 0:
                                pitches = sorted({n.pitch.midi for n in el.notes})
                                measure_events.append((part_name, offset, f"note_{'-'.join(map(str, pitches))}"))
                        elif isinstance(el, note.Rest):
                            measure_events.append((part_name, offset, "rest"))

        # Sort the measurement events by increasing offset
        measure_events.sort(key=lambda x: x[1])
        measure_events.sort(key=lambda x: x[0])

        events.extend(measure_events)

    return events



def process_buffer(buf):
    result = []
    i = 0
    while i < len(buf):
        current = buf[i]
        part_name, offset, ev = current
        group = [current]
        i += 1
        # Group together all the following events that have the same part and the same offset
        while i < len(buf) and buf[i][0] == part_name and buf[i][1] == offset:
            group.append(buf[i])
            i += 1
        if len(group) == 1:
            # Only one element, we keep it as is
            result.append(group[0])
        else:
            # Several elements, processing
            notes = [e for e in group if e[2] != "rest"]
            rests = [e for e in group if e[2] == "rest"]

            if len(notes) == 0:
                # Only silence, we'll keep just one
                result.append(rests[0])
            elif len(notes) == 1:
                # Only one note, we'll keep this note.
                result.append(notes[0])
            else:
                # Several notes, we merge into a chord
                all_pitches = []
                for _, _, note_str in notes:
                    pitches_str = note_str[5:]
                    all_pitches.extend(map(int, pitches_str.split("-")))
                unique_pitches = sorted(set(all_pitches))
                chord_str = f"note_{'-'.join(map(str, unique_pitches))}"
                result.append((part_name, offset, chord_str))
                
    return result



def merge_events_simple(events):
    merged = []
    buffer = []  
    measure_number = 1  # measurement counter
    
    for event in events:
        try:
            if event[0] == "barline":
                if buffer:
                    merged.extend(process_buffer(buffer))
                    buffer = []
                merged.append(event)
                measure_number += 1 
            elif event[0] == "time_signature":
                merged.append(event)
            else:
                buffer.append(event)
        except Exception as e:
            print(f"Erreur dans la mesure {measure_number} avec event : {event}")
            raise e

    # Process the last measurement if there are any events remaining
    if buffer:
        try:
            merged.extend(process_buffer(buffer))
        except Exception as e:
            print(f"Erreur dans la dernière mesure ({measure_number}) avec buffer : {buffer}")
            raise e

    return merged


def decode_to_score(events, n_hand, i=None):

    score = stream.Score()

    right_hand = stream.Part()
    right_hand.insert(0, instrument.Piano())
    right_hand.partName = "main_droite"

    left_hand = stream.Part()
    left_hand.insert(0, instrument.Piano())
    left_hand.partName = "main_gauche"
    left_hand.insert(0, clef.BassClef())

    # Take the first time signature encountered
    first_ts = next((e[1] for e in events if e[0] == "time_signature"), "4/4")
    current_time_signature = meter.TimeSignature(first_ts)
    measure_duration = current_time_signature.barDuration.quarterLength

    measure_number = 0
    measure_rh = stream.Measure()
    measure_lh = stream.Measure()

    measure_rh.insert(0, current_time_signature)
    measure_lh.insert(0, current_time_signature)

    buffer = []

    for e in events:
        if e[0] == "barline":
            if buffer:
                events_by_hand = {"main_droite": [], "main_gauche": []}
                for part_name, offset, label in buffer:
                    events_by_hand[part_name].append((offset, label))

                for hand, hand_events in events_by_hand.items():
                    hand_events.sort(key=lambda x: x[0])
                    for i, (offset, label) in enumerate(hand_events):
                        if i + 1 < len(hand_events):
                            duration = Fraction(hand_events[i+1][0] - offset)
                        else:
                            duration = Fraction(measure_duration - offset)

                        if label.startswith("note_"):
                            pitches = [int(p) for p in label[5:].split("-")]
                            pitch_objs = [pitch.Pitch(midi=p) for p in pitches]
                            el = chord.Chord(pitch_objs, quarterLength=duration) if len(pitch_objs) > 1 else note.Note(pitch_objs[0], quarterLength=duration)
                        
                        elif label == "rest":
                            el = note.Rest(quarterLength=duration)
                        else:
                            continue

                        if hand == "main_droite":
                            measure_rh.insert(offset, el)
                        else:
                            measure_lh.insert(offset, el)

                buffer = []

            measure_number += 1
            measure_rh.number = measure_number
            measure_lh.number = measure_number
            if len(measure_rh.notesAndRests) > 0:
                right_hand.append(measure_rh)
            if len(measure_lh.notesAndRests) > 0:
                left_hand.append(measure_lh)


            measure_rh = stream.Measure()
            measure_lh = stream.Measure()

        elif e[0] == "time_signature":
            ts_str = e[1]
            current_time_signature = meter.TimeSignature(ts_str)
            measure_duration = current_time_signature.barDuration.quarterLength
            measure_rh.insert(0, current_time_signature)
            measure_lh.insert(0, current_time_signature)

        else:
            # format = (part_name, offset, label)
            part_name, offset, label = e
            buffer.append((part_name, offset, label))

    # Latest measure
    if buffer:
        events_by_hand = {"main_droite": [], "main_gauche": []}
        for part_name, offset, label in buffer:
            events_by_hand[part_name].append((offset, label))

        for hand, hand_events in events_by_hand.items():
            hand_events.sort(key=lambda x: x[0])
            for i, (offset, label) in enumerate(hand_events):
                if i + 1 < len(hand_events):
                    duration = Fraction(hand_events[i+1][0] - offset)
                else:
                    duration = Fraction(measure_duration - offset)

                if label.startswith("note_"):
                    pitches = [int(p) for p in label[5:].split("-")]
                    el = chord.Chord(pitches, quarterLength=duration) if len(pitches) > 1 else note.Note(pitches[0], quarterLength=duration)
                elif label == "rest":
                    el = note.Rest(quarterLength=duration)
                else:
                    continue

                if hand == "main_droite":
                    measure_rh.insert(offset, el)
                else:
                    measure_lh.insert(offset, el)

        measure_number += 1
        measure_rh.number = measure_number
        measure_lh.number = measure_number
        right_hand.append(measure_rh)
        left_hand.append(measure_lh)

    score.insert(0, right_hand)
    if n_hand == 2:
        score.insert(0, left_hand)
        group = layout.StaffGroup([right_hand, left_hand], name="Piano", abbreviation="Pno.", symbol="brace")
        score.insert(0, group)

    return score

def parse_time_signature(ts_str):
    num, denom = map(int, ts_str.split("/"))
    return num * (4.0 / denom)


def events_to_tokens2(events):
    tokens = []
    current_ts = -1
    measure_duration = -1
    inside_measure_right = False # flag: are we in a measure after a barline?
    inside_measure_left = False

    i = 0
    while i < len(events):
        e = events[i]

        if e[0] == "barline":
            tokens.append("barline")
            inside_measure_right = False 
            inside_measure_left = False
            i += 1
            continue

        elif e[0] == "time_signature":
            current_ts = e[1]
            measure_duration = parse_time_signature(current_ts)
            tokens.append(f"time_signature_{current_ts}")
            i += 1
            continue

        for main in ["main_droite", "main_gauche"]:
            if e[0] == main:
                offset, label = e[1], e[2]

                # Choose the correct flag according to the hand
                if main == "main_droite":
                    if not inside_measure_right:
                        tokens.append(main)
                        inside_measure_right = True
                else:  # main_gauche
                    if not inside_measure_left:
                        tokens.append(main)
                        inside_measure_left = True


                # Split the notes if there are multiple ones separated by "-"
                if "-" in label:
                    parts = label.split("-")
                    if parts[0].startswith("note_"):
                        base = "note_"
                        for p in parts:
                            # if p already contains "note_", keep it as is, otherwise add base
                            tokens.append(p if p.startswith("note_") else f"{base}{p}")
                    else:
                        tokens.extend(parts)
                else:
                    tokens.append(label)

                
                # Look for the next event to calculate the duration
                if i + 1 < len(events):
                    next_event = events[i + 1]

                    if next_event[0] == main:
                        next_offset = next_event[1]
                        duration = Fraction(next_offset - offset).limit_denominator(64)
                    else: #next_event[0] == "barline"
                        duration = Fraction(measure_duration - offset).limit_denominator(64)
                else:
                    duration = Fraction(measure_duration - offset).limit_denominator(64)

                tokens.append(f"duration_{duration}")
                i += 1
                continue

    return tokens

def tokens_to_events2(tokens):
    events = []
    measure_count = 0
    offset_right = 0
    offset_left = 0
    current_ts = -1
    measure_duration = -1
    right_or_left = None
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        if tok == "barline":
            measure_count += 1
            events.append(("barline", f"measure_{measure_count}"))
            offset_right = 0.0
            offset_left = 0.0
            i += 1
            continue

        elif tok.startswith("time_signature_"):
            current_ts = tok.replace("time_signature_", "")
            measure_duration = parse_time_signature(current_ts)
            events.append(("time_signature", current_ts))
            i += 1
            continue

        elif tok == "main_droite":
            right_or_left = "right"
            i += 1
            continue
        
        elif tok == "main_gauche":
            right_or_left = "left"
            i += 1
            continue

        else:
            # Start a chord with the current token
            chord_notes = [tok]
            j = i + 1

            # Group all consecutive notes
            while j < len(tokens) and tokens[j].startswith("note_"):
                chord_notes.append(tokens[j])
                j += 1

            # Retrieve the duration if it exists
            if j < len(tokens) and tokens[j].startswith("duration_"):
                duration_token = tokens[j].replace("duration_", "")
                duration = float(Fraction(duration_token))
                j += 1 
            else:
                raise ValueError(f"Durée manquante pour le chord commençant par {chord_notes[0]}")

            if len(chord_notes) == 1:
                chord_label = chord_notes[0]
            else:
                chord_label = "note_" + "-".join([n.replace("note_", "") for n in chord_notes])

            if right_or_left == "right":
                events.append(("main_droite", offset_right, chord_label))
                offset_right += duration
            else:
                events.append(("main_gauche", offset_left, chord_label))
                offset_left += duration
            
            i = j # pass all processed tokens (notes + duration)
            continue
        
    return events