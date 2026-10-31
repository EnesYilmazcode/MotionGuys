import assert from 'node:assert/strict';
import { test } from 'node:test';
import { describeRound, ranking, roundStateAt, shotAt, spectatorShots, tileKey, topRunners, type SpleefMeta } from '../src/game/spleef.ts';

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

test('only runners on the top floor who are on their feet count as top', () => {
  assert.deepEqual(topRunners(roundStateAt(meta, 3.5)), [1, 2]);
  assert.deepEqual(topRunners(roundStateAt(meta, 5.5)), [2]);
});

test('the camera cuts away shortly after a fall and ends on the winner', () => {
  const shots = spectatorShots(meta, 8, 100);
  assert.equal(shots[shots.length - 1].runner, 2);
  assert.ok(shots[shots.length - 1].start <= 7);
  for (const shot of shots.slice(0, -1)) {
    const fall = meta.events.find(event => event.t >= shot.start && event.runner === shot.runner && (event.type === 'fall' || event.type === 'eliminated'));
    assert.ok(fall, 'a shot with a 100 s hold only ends because its runner fell');
    assert.equal(shotAt(shots, fall.t).runner, shot.runner, 'the fall itself stays on screen');
  }
});

test('the same seed always gives the same shots', () => {
  assert.deepEqual(spectatorShots(meta, 8), spectatorShots(meta, 8));
});

test('events read as plain sentences', () => {
  assert.equal(describeRound(meta.events[2], meta), 'Ada fell through');
  assert.equal(describeRound(meta.events[0], meta), null);
});
