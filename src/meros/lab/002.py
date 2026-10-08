from itertools import combinations
from src.meros import media, metadata, Individual
from src.meros.domain.storage import TrackSelection
from src.meros.engine import pipeline
from .utils import compare_representation

def compare_tracks(a: TrackSelection, b: TrackSelection) -> None:
    print()
    print(f'Comparing {a}{a.video_id}/{a.track_id}@{a.reference_frame} with {b.video_id}/{b.track_id}@{b.reference_frame}')
    source_reference = media.read_masked_crop(a.video_id, a.track_id, a.reference_frame)
    target_reference = media.read_masked_crop(b.video_id, b.track_id, b.reference_frame)
    source_composite = media.read_composite(a.video_id, a.track_id, a.reference_frame, selection_id=a.selection_id)
    target_composite = media.read_composite(b.video_id, b.track_id, b.reference_frame, selection_id=b.selection_id)
    source_enhanced = media.read_composite_enhanced(a.video_id, a.track_id, a.reference_frame, selection_id=a.selection_id)
    target_enhanced = media.read_composite_enhanced(b.video_id, b.track_id, b.reference_frame, selection_id=b.selection_id)
    compare_representation('reference', source_reference, target_reference)
    compare_representation('composite', source_composite, target_composite)
    compare_representation('enhanced', source_enhanced, target_enhanced)

def compare_same_individuals(individual: Individual) -> None:
    for a, b in combinations(individual.tracks, 2):
        compare_tracks(a, b)

def compare_different_individuals(a: Individual, b: Individual) -> None:
    for source_track in a.tracks:
        for target_track in b.tracks:
            compare_tracks(source_track, target_track)

class Exp:

    def run(self) -> None:
        pipeline.run('enhanced_composites')
        individuals = metadata.read_individuals()
        for individual in individuals:
            tracks = individual.tracks
            for track in tracks:
                print(f'Individual {individual.individual_id} appears in video {track.video_id}/{track.track_id}')
        for individual in individuals:
            compare_same_individuals(individual)
        for a, b in combinations(individuals, 2):
            compare_different_individuals(a, b)
if __name__ == '__main__':
    exp = Exp()
    exp.run()
