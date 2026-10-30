import { actionAt, course, type Action } from './course.ts';

export const STEP = 1 / 60;
const colors = ['#b8f153', '#fb7ca8', '#8faaff', '#ffb25b', '#65dce5', '#ce96ff', '#f0e579', '#ff7e69'];
export interface Runner {
  id: number; name: string; color: string; x: number; z: number;
  speed: number; action: Action; finishedAt: number | null;
}

export class Race {
  time = 0;
  runners: Runner[];
  readonly seed: number;

  constructor(seed = 42) {
    this.seed = seed;
    let value = seed >>> 0;
    const random = () => { value = (Math.imul(value, 1664525) + 1013904223) >>> 0; return value / 4294967296; };
    this.runners = colors.map((color, id) => ({
      id, name: `Runner ${id + 1}`, color, x: -5.25 + id * 1.5, z: 2,
      speed: 3.3 + random() * 0.9, action: 'run', finishedAt: null,
    }));
  }

  step() {
    if (this.complete) return;
    this.time += STEP;
    for (const runner of this.runners) {
      if (runner.finishedAt !== null) continue;
      runner.action = actionAt(runner.z);
      const pace = runner.action === 'duck' ? 0.55 : runner.action === 'vault' ? 0.7 : 1;
      const next = runner.z + runner.speed * pace * STEP;
      if (next >= course.finish) {
        runner.finishedAt = this.time - STEP + (course.finish - runner.z) / (runner.speed * pace);
        runner.z = course.finish;
        runner.action = 'run';
      } else runner.z = next;
    }
  }

  get complete() { return this.runners.every(r => r.finishedAt !== null); }
  get standings() {
    return [...this.runners].sort((a, b) => {
      if (a.finishedAt !== null && b.finishedAt !== null) return a.finishedAt - b.finishedAt;
      if (a.finishedAt !== null) return -1;
      if (b.finishedAt !== null) return 1;
      return b.z - a.z;
    });
  }
}
