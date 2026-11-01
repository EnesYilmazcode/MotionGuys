import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { actionShots, brainStateAt, describeHex, hexCenter, hexKey, tileTimes, tileWhiteness, type HexMeta } from '../src/game/hex.ts';

const rounds = readdirSync('public/hex').filter(name => name.endsWith('.json')).map(name => JSON.parse(readFileSync(`public/hex/${name}`, 'utf8')) as HexMeta);

test('a tile holds its color, whitens, then vanishes', () => {
  const entry = { touched: 10, gone: 10.9 };
  assert.equal(tileWhiteness(9, entry, 0.9), 0);
  assert.equal(tileWhiteness(10.2, entry, 0.9), 0, 'the first 47% of the timer looks untouched');
  assert.ok(tileWhiteness(10.7, entry, 0.9) > 0.5);
  assert.equal(tileWhiteness(10.89, entry, 0.9), 1);
  assert.equal(tileWhiteness(10.9, entry, 0.9), -1);
  assert.equal(tileWhiteness(99, undefined, 0.9), 0, 'an untouched tile never changes');
});

test('the honeycomb has no gaps or overlaps', () => {
  const radius = 1.2;
  const [x0, y0] = hexCenter(0, 0, radius);
  for (const [q, r] of [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]]) {
    const [x, y] = hexCenter(q, r, radius);
    assert.ok(Math.abs(Math.hypot(x - x0, y - y0) - Math.sqrt(3) * radius) < 1e-9);
  }
});

test('a fall line is the same on every replay', () => {
  const event = { t: 12.34, type: 'fall', runner: 1 };
  assert.equal(describeHex(event, ['A', 'B']), describeHex(event, ['A', 'B']));
  assert.match(describeHex(event, ['A', 'B'])!, /B/);
  assert.equal(describeHex({ t: 1, type: 'tile_touch', runner: 1 }, ['A', 'B']), null);
});

test('the camera cuts just before a knockdown, and not constantly', () => {
  const meta = { events: [
    { t: 10, type: 'knock', runner: 2, by: 1 }, { t: 11, type: 'knock', runner: 3, by: 1 }, { t: 20, type: 'knock', runner: 4, by: 1 }, { t: 30, type: 'winner', runner: 5 },
  ] } as unknown as HexMeta;
  const shots = actionShots(meta, [{ start: 0, runner: 0 }, { start: 10, runner: 7 }, { start: 15, runner: 6 }], 0.7, 3.5);
  assert.deepEqual(shots, [{ start: 0, runner: 0 }, { start: 9.3, runner: 2 }, { start: 15, runner: 6 }, { start: 19.3, runner: 4 }]);
});

test('every physics event has a line', () => {
  assert.match(describeHex({ t: 1, type: 'knock', runner: 0, by: 1 }, ['A', 'B'])!, /A/);
  assert.match(describeHex({ t: 1, type: 'knock', runner: 0, by: 1 }, ['A', 'B'])!, /B/);
});

test('the panel replays tile counts and gaits from the log', () => {
  const meta = { runners: [{}, {}], events: [
    { t: 1, type: 'gait', runner: 0, gait: 2 }, { t: 2, type: 'tile_touch', runner: 0 }, { t: 3, type: 'gait', runner: 0, gait: 0 }, { t: 4, type: 'tile_touch', runner: 1 },
  ] } as unknown as HexMeta;
  assert.deepEqual(brainStateAt(meta, 2.5), { tiles: [1, 0], gait: [2, null] });
  assert.deepEqual(brainStateAt(meta, 9), { tiles: [1, 1], gait: [0, null] });
});

test('recordings exist', () => assert.ok(rounds.length >= 6));

for (const meta of rounds) {
  test(`seed ${meta.seed}: every event fits the arena`, () => {
    const { arena } = meta;
    const known = new Set(arena.tiles.flatMap((tiles, layer) => tiles.map(([q, r]) => hexKey(layer, q, r))));
    for (const event of meta.events) {
      if (event.type !== 'tile_touch' && event.type !== 'tile_gone') continue;
      assert.ok(known.has(hexKey(event.layer as number, event.q as number, event.r as number)));
    }
    for (const entry of tileTimes(meta).values()) {
      if (entry.gone === Infinity) continue;
      assert.ok(entry.gone - entry.touched >= arena.touchSeconds - 0.01, 'a tile never vanishes early');
    }
    assert.equal(meta.events.filter(event => event.type === 'winner').length, 1);
  });

  test(`seed ${meta.seed}: floors grow downward and nobody outlasts the winner`, () => {
    const { radii } = meta.arena;
    radii.slice(1).forEach((radius, i) => assert.ok(radius > radii[i]));
    meta.arena.tiles.slice(1).forEach((tiles, i) => assert.ok(tiles.length > meta.arena.tiles[i].length));
    const win = meta.events.find(event => event.type === 'winner')!;
    // A winner can already be falling when the last rival hits the slime. Going in last still wins.
    const winnerOut = meta.runners[win.runner as number].eliminatedAt ?? Infinity;
    meta.runners.forEach((runner, index) => {
      if (index === win.runner) return;
      assert.ok(runner.eliminatedAt !== null, `${runner.name} is out`);
      assert.ok(runner.eliminatedAt! <= winnerOut);
    });
  });
}
