import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Race } from '../src/game/race.ts';
import { course } from '../src/game/course.ts';

function run(seed: number) {
  const race = new Race(seed);
  const actions = new Set<string>();
  for (let step = 0; step < 60 * 120 && !race.complete; step++) {
    race.step();
    race.runners.forEach(r => actions.add(r.action));
  }
  return { race, actions };
}

test('a seeded race repeats the same finish times and order', () => {
  assert.deepEqual(run(42).race.standings, run(42).race.standings);
  assert.notDeepEqual(run(42).race.standings, run(43).race.standings);
});

test('all runners traverse the course actions and stop at the finish', () => {
  const { race, actions } = run(42);
  assert.equal(race.complete, true);
  assert.deepEqual([...actions].sort(), ['duck', 'jump', 'run', 'vault']);
  for (const runner of race.runners) {
    assert.equal(runner.z, course.finish);
    assert.ok(runner.finishedAt !== null && runner.finishedAt > 0);
  }
  assert.deepEqual(race.standings.map(r => r.finishedAt), race.standings.map(r => r.finishedAt).sort((a, b) => a! - b!));
  const snapshot = JSON.stringify(race.runners);
  race.step();
  assert.equal(JSON.stringify(race.runners), snapshot);
});
