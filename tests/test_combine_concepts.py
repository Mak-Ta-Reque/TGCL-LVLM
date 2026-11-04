"""
Tests for combine_concepts filtering logic. Ensures only concepts where ALL
image_grounding_predictions are positive (none match negative conditions) are
included.
"""

import os
import shutil
import tempfile

import torch

from tgcl.combine_concepts import combine_concepts
from tgcl.combine_concepts import eligible_indices_by_threshold
from tgcl.combine_concepts import has_all_positive_predictions
from tgcl.combine_concepts import is_negative_prediction

# ── Unit tests for is_negative_prediction ──────────────────────────────────


def test_is_negative_prediction():
    # Negative cases
    assert is_negative_prediction('no') is True, "'no' should be negative"
    assert (
        is_negative_prediction('No') is True
    ), "'No' should be negative (case-insensitive)"
    assert (
        is_negative_prediction('NO') is True
    ), "'NO' should be negative (case-insensitive)"
    assert (
        is_negative_prediction('no_object') is True
    ), "'no_object' should be negative"
    assert (
        is_negative_prediction('not_found') is True
    ), "'not_found' should be negative"
    assert is_negative_prediction('unk') is True, "'unk' should be negative"
    assert (
        is_negative_prediction('unknown') is True
    ), "'unknown' should be negative"
    assert is_negative_prediction('thing') is True, "'thing' should be negative"
    assert (
        is_negative_prediction('thing_123') is True
    ), "'thing_123' should be negative"
    assert is_negative_prediction('nc') is True, "'nc' should be negative"
    assert (
        is_negative_prediction('nc_label') is True
    ), "'nc_label' should be negative"

    # Positive cases — should NOT be flagged as negative
    assert is_negative_prediction('cat') is False, "'cat' should be positive"
    assert is_negative_prediction('dog') is False, "'dog' should be positive"
    assert (
        is_negative_prediction('normal') is False
    ), "'normal' should be positive (not exact 'no')"
    assert (
        is_negative_prediction('nose') is False
    ), "'nose' should be positive (not exact 'no')"
    assert (
        is_negative_prediction('north') is False
    ), "'north' should be positive"
    assert (
        is_negative_prediction('notice') is False
    ), "'notice' should be positive"
    assert is_negative_prediction('car') is False, "'car' should be positive"
    assert (
        is_negative_prediction('person') is False
    ), "'person' should be positive"

    print('  PASSED: test_is_negative_prediction')


# ── Unit tests for has_all_positive_predictions ────────────────────────────


def test_has_all_positive_predictions():
    # All positive
    assert has_all_positive_predictions(['cat', 'dog', 'car']) is True
    # One negative
    assert has_all_positive_predictions(['cat', 'no_object', 'car']) is False
    # All negative
    assert has_all_positive_predictions(['no', 'unk', 'nc']) is False
    # Mixed with 'no' exact
    assert has_all_positive_predictions(['cat', 'no', 'dog']) is False
    # Empty list -> False (no evidence of positive)
    assert has_all_positive_predictions([]) is False
    # Single positive
    assert has_all_positive_predictions(['bird']) is True
    # Single negative
    assert has_all_positive_predictions(['not_found']) is False
    # 'normal' should be positive (NOT caught by 'no' prefix)
    assert has_all_positive_predictions(['normal', 'nose', 'cat']) is True

    print('  PASSED: test_has_all_positive_predictions')


# ── Unit tests for eligible_indices_by_threshold ───────────────────────────


def test_eligible_indices_by_threshold():
    predictions = [
        ['cat', 'dog', 'bird'],  # idx 0: ALL positive -> eligible
        ['cat', 'no_object', 'bird'],  # idx 1: has negative -> NOT eligible
        ['car', 'truck', 'plane'],  # idx 2: ALL positive -> eligible
        ['no', 'unk', 'nc'],  # idx 3: ALL negative -> NOT eligible
        ['person', 'thing_1', 'bike'],  # idx 4: has negative -> NOT eligible
    ]
    eligible = eligible_indices_by_threshold(predictions)
    assert eligible == [0, 2], f'Expected [0, 2], got {eligible}'

    # All negative -> no eligible
    all_neg = [['no', 'unk'], ['not_found', 'nc']]
    assert eligible_indices_by_threshold(all_neg) == []

    # All positive -> all eligible
    all_pos = [['cat', 'dog'], ['bird', 'fish']]
    assert eligible_indices_by_threshold(all_pos) == [0, 1]

    # Empty list
    assert eligible_indices_by_threshold([]) == []

    # Single concept, all positive
    assert eligible_indices_by_threshold([['cat', 'dog']]) == [0]

    # Single concept, has negative
    assert eligible_indices_by_threshold([['cat', 'no']]) == []

    print('  PASSED: test_eligible_indices_by_threshold')


# ── Integration test with dummy .pth files ─────────────────────────────────


def create_dummy_pth(filepath, predictions, n_features=4, n_samples=3):
    """Create a dummy .pth file with given predictions."""
    n_concepts = len(predictions)
    data = {
        'concepts': torch.randn(n_concepts, n_features),
        'activations': torch.randn(n_samples, n_concepts),
        'decomposition_method': 'pca',
        'text_grounding': [f'text_{i}' for i in range(n_concepts)],
        'image_grounding_paths': [
            [f'img_{i}_{j}.png' for j in range(3)] for i in range(n_concepts)
        ],
        'image_concept_names': [f'concept_{i}' for i in range(n_concepts)],
        'analysis_model': 'test_model',
        'image_grounding_predictions': predictions,
        'image_grounding_bboxes': [[(0, 0, 10, 10)] for _ in range(n_concepts)],
        'image_grounding_masks': [None for _ in range(n_concepts)],
    }
    torch.save(data, filepath)
    return data


def test_combine_concepts_integration():
    """
    Integration test: create dummy .pth files and verify that combine_concepts
    only includes concepts where ALL predictions are positive.
    """
    tmpdir = tempfile.mkdtemp(prefix='test_combine_')
    try:
        # File 1: concept 0 is all positive, concept 1 has a negative
        create_dummy_pth(
            os.path.join(tmpdir, 'model_a.pth'),
            predictions=[
                ['cat', 'dog', 'bird'],  # idx 0: ALL positive -> INCLUDED
                ['cat', 'no_object', 'bird'],  # idx 1: has negative -> EXCLUDED
            ],
        )

        # File 2: concept 0 has negative, concept 1 is all positive
        create_dummy_pth(
            os.path.join(tmpdir, 'model_b.pth'),
            predictions=[
                ['not_found', 'unk', 'nc'],  # idx 0: ALL negative -> EXCLUDED
                ['car', 'truck', 'plane'],  # idx 1: ALL positive -> INCLUDED
            ],
        )

        # File 3: ALL concepts have negatives -> entire file should be SKIPPED
        create_dummy_pth(
            os.path.join(tmpdir, 'model_c.pth'),
            predictions=[
                ['no', 'thing_x', 'nc'],  # idx 0: ALL negative
                ['no_object', 'unk', 'not_found'],  # idx 1: ALL negative
            ],
        )

        # File 4: ALL concepts are positive -> both included
        create_dummy_pth(
            os.path.join(tmpdir, 'model_d.pth'),
            predictions=[
                ['person', 'bicycle', 'car'],  # idx 0: ALL positive -> INCLUDED
                ['horse', 'sheep', 'cow'],  # idx 1: ALL positive -> INCLUDED
            ],
        )

        result, negative = combine_concepts(tmpdir)

        # Expected: 4 concepts total (1 from a, 1 from b, 0 from c, 2 from d)
        n_concepts = result['concepts'].shape[0]
        assert n_concepts == 4, f'Expected 4 concepts, got {n_concepts}'

        # Verify all predictions in the result are positive
        for i, preds in enumerate(result['image_grounding_predictions']):
            if preds is not None:
                for p in preds:
                    assert not is_negative_prediction(p), (
                        f"Concept {i} has negative prediction '{p}' — should "
                        'have been filtered!'
                    )

        # Verify concept names
        names = result['concept_names']
        assert len(names) == 4, f'Expected 4 concept names, got {len(names)}'

        # The filtered-out concepts land in the negative bank instead of
        # being discarded: 1 from a, 1 from b, 2 from c.
        n_negative = negative['concepts'].shape[0]
        assert (
            n_negative == 4
        ), f'Expected 4 negative concepts, got {n_negative}'
        assert len(negative['concept_names']) == 4
        # Every negative concept has at least one negative prediction
        for preds in negative['image_grounding_predictions']:
            assert preds is None or any(
                is_negative_prediction(p) for p in preds
            )

        print('  PASSED: test_combine_concepts_integration')

    finally:
        shutil.rmtree(tmpdir)


def test_no_purity_filter_keeps_every_direction():
    """Ratio 0 (pooled baselines): directions with negative predictions and
    directions no sample is assigned to are all kept."""
    tmpdir = tempfile.mkdtemp(prefix='test_combine_all_')
    try:
        path = os.path.join(tmpdir, 'model_pooled.pth')
        create_dummy_pth(
            path,
            predictions=[['cat', 'dog'], ['no', 'unk'], ['car', 'no_object']],
        )
        data = torch.load(path, weights_only=False)
        data['direction_positive_ratio'] = [1.0, 0.0, None]
        data['direction_num_assigned'] = [5, 3, 0]  # the last one: no sample
        torch.save(data, path)
        result, negative = combine_concepts(tmpdir, clean_example_ratio=0.0)
        assert result['concepts'].shape[0] == 3
        assert negative['concepts'].numel() == 0
    finally:
        shutil.rmtree(tmpdir)


def test_combine_concepts_all_negative():
    """If ALL files have only negative predictions, result should be empty."""
    tmpdir = tempfile.mkdtemp(prefix='test_combine_neg_')
    try:
        create_dummy_pth(
            os.path.join(tmpdir, 'model_neg.pth'),
            predictions=[
                ['no', 'unk', 'nc'],
                ['not_found', 'thing_x', 'no_object'],
            ],
        )

        result, negative = combine_concepts(tmpdir)
        assert result['concepts'].numel() == 0, 'Expected empty concepts tensor'
        assert len(result['image_grounding_predictions']) == 0
        # ...but nothing is lost: both concepts land in the negative bank
        assert negative['concepts'].shape[0] == 2
        print('  PASSED: test_combine_concepts_all_negative')

    finally:
        shutil.rmtree(tmpdir)


def test_edge_case_normal_nose_not_flagged():
    """
    Regression test: words like 'normal' and 'nose' should NOT be flagged as
    negative. Previously startswith('no') was too broad.
    """
    tmpdir = tempfile.mkdtemp(prefix='test_combine_edge_')
    try:
        create_dummy_pth(
            os.path.join(tmpdir, 'model_edge.pth'),
            predictions=[
                ['normal', 'nose', 'north'],  # ALL positive (not exact 'no')
            ],
        )

        result, _ = combine_concepts(tmpdir)
        assert result['concepts'].shape[0] == 1, (
            'Expected 1 concept (normal/nose/north are positive), got '
            f"{result['concepts'].shape[0]}"
        )
        print('  PASSED: test_edge_case_normal_nose_not_flagged')

    finally:
        shutil.rmtree(tmpdir)


def test_combine_concepts_logging():
    """Verify that combine_concepts prints skip/include stats and summary."""
    tmpdir = tempfile.mkdtemp(prefix='test_combine_log_')
    try:
        # 2 concepts: one positive, one negative
        create_dummy_pth(
            os.path.join(tmpdir, 'model_log.pth'),
            predictions=[
                ['cat', 'dog'],  # positive -> INCLUDED
                ['no', 'unk'],  # negative -> SKIPPED
            ],
        )

        # Capture stdout
        from contextlib import redirect_stdout
        import io

        f = io.StringIO()
        with redirect_stdout(f):
            result, _ = combine_concepts(tmpdir)

        output = f.getvalue()

        # Verify summary stats appear
        assert 'COMBINE CONCEPTS SUMMARY' in output, 'Missing summary header'
        assert (
            'Negative bank:          1' in output
        ), f'Should report 1 negative-bank concept, got: {output}'
        assert (
            'Positive bank:          1' in output
        ), f'Should report 1 positive-bank concept, got: {output}'
        assert 'NEGATIVE' in output, 'Should show NEGATIVE for negative concept'
        assert 'POSITIVE' in output, 'Should show POSITIVE for positive concept'
        assert (
            'negative predictions:' in output
        ), 'Should show which predictions were negative'

        # Verify the actual result is correct too
        assert (
            result['concepts'].shape[0] == 1
        ), f"Expected 1 included concept, got {result['concepts'].shape[0]}"

        print('  PASSED: test_combine_concepts_logging')

    finally:
        shutil.rmtree(tmpdir)


# ── Run all tests ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Running unit tests...')
    test_is_negative_prediction()
    test_has_all_positive_predictions()
    test_eligible_indices_by_threshold()

    print('\nRunning integration tests...')
    test_combine_concepts_integration()
    test_combine_concepts_all_negative()
    test_edge_case_normal_nose_not_flagged()
    test_combine_concepts_logging()

    print('\n✓ All tests passed!')
