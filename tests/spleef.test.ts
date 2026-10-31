import assert from 'node:assert/strict';
import { test } from 'node:test';
import { describeRound, ranking, roundStateAt, tileKey, type SpleefMeta } from '../src/game/spleef.ts';

const runner = (name: string) => ({ name, color: '#fff', gear: 'cap', bulk: 1, eliminatedAt: null });
const meta: SpleefMeta = {
  seed: 1, fps: 30, frames: 2, joints: ['Hips'],
  arena: { layers: 3, layerHeight: 6, tile: 1.25, tiles: 16, half: 10, dropSeconds: 1 },
  runners: [runner('Ada'), runner('Bo'), runner('Cleo')],
  events: [
    { t: 1, type: 'tile_shake', runner: 0, layer: 0, column: 2, row: 3 },
    { t: 2, type: 'tile_drop', runner: null, layer: 0, column: 2, row: 3 },
    { t: 3, type: 'fall', runner: 0, layer: 0, cause: 'hole' },
    { t: 4, type: 'land', runner: 0, layer: 1 },
    { t: 5, type: 'eliminated', runner: 1 },
    { t: 6, type: 'eliminated', runner: 0 },
    { t: 7, type: 'winner', runner: 2 },
  ],
};

test('tiles shake, then drop', () => {
  const key = tileKey(0, 2, 3);
  assert.ok(roundStateAt(meta, 1.5).shaking.has(key));
  const later = roundStateAt(meta, 2.5);
  assert.equal(later.shaking.has(key), false);
  assert.equal(later.gone.get(key), 2);
});

test('falling moves a runner down a floor', () => {
  assert.equal(roundStateAt(meta, 3.5).layer[0], 0);
  assert.equal(roundStateAt(meta, 4.5).layer[0], 1);
  assert.equal(roundStateAt(meta, 3.5).falls[0], 1);
});

test('the winner ranks first, then whoever lasted longest', () => {
  assert.deepEqual(ranking(meta, roundStateAt(meta, 8)).map(r => r.name), ['Cleo', 'Ada', 'Bo']);
});

test('runners still in rank ahead of runners who are out', () => {
  assert.equal(ranking(meta, roundStateAt(meta, 5.5))[2].name, 'Bo');
});

test('events read as plain sentences', () => {
  assert.equal(describeRound(meta.events[2], meta), 'Ada fell through');
  assert.equal(describeRound(meta.events[0], meta), null);
});
