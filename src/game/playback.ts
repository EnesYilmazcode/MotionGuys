export interface RaceEvent { t: number; type: string; runner: number | null; [detail: string]: unknown; }
export interface Sweeper { id: string; x: number; z: number; radius: number; length: number; speed: number; angle: number; action: string; }
export interface RaceMeta {
  seed: number;
  fps: number;
  frames: number;
  joints: string[];
  course: {
    halfWidth: number;
    finishX: number;
    doorRows: { x: number; real: number[] }[];
    doorCount: number;
    sweepers: Sweeper[];
    postRadius: number;
    tiles: { x: number; size: number; rows: number; columns: number };
  };
  runners: { name: string; color: string; finishedAt: number | null }[];
  events: RaceEvent[];
}
export type Vec3 = [number, number, number];

/** Eight runners of recorded joints: hips as float32 meters, then joints as int16 millimeters from the hips. */
export class Recording {
  readonly hips: Float32Array;
  readonly joints: Int16Array;
  readonly jointCount: number;
  readonly runnerCount: number;
  readonly meta: RaceMeta;

  constructor(meta: RaceMeta, buffer: ArrayBuffer) {
    this.meta = meta;
    this.runnerCount = meta.runners.length;
    this.jointCount = meta.joints.length;
    const hipsLength = meta.frames * this.runnerCount * 3;
    const jointsLength = hipsLength * this.jointCount;
    const expected = hipsLength * 4 + jointsLength * 2;
    if (buffer.byteLength !== expected) throw new Error(`Recording is ${buffer.byteLength} bytes, expected ${expected}`);
    this.hips = new Float32Array(buffer, 0, hipsLength);
    this.joints = new Int16Array(buffer, hipsLength * 4, jointsLength);
  }

  get duration() { return (this.meta.frames - 1) / this.meta.fps; }

  /** Whether the runner is still being recorded at this frame. */
  present(frame: number, runner: number) {
    return !Number.isNaN(this.hips[(frame * this.runnerCount + runner) * 3]);
  }

  /** World position of a joint at a fractional time, or null once the runner has left the race. */
  joint(time: number, runner: number, joint: number, out: Vec3): Vec3 | null {
    const position = Math.min(Math.max(time, 0) * this.meta.fps, this.meta.frames - 1);
    const a = Math.floor(position);
    const b = Math.min(a + 1, this.meta.frames - 1);
    if (!this.present(a, runner)) return null;
    const blend = this.present(b, runner) ? position - a : 0;
    for (let i = 0; i < 3; i++) out[i] = this.mix(a, b, blend, runner, joint, i);
    return out;
  }

  private mix(a: number, b: number, blend: number, runner: number, joint: number, axis: number) {
    return this.value(a, runner, joint, axis) * (1 - blend) + this.value(b, runner, joint, axis) * blend;
  }

  private value(frame: number, runner: number, joint: number, axis: number) {
    const slot = frame * this.runnerCount + runner;
    return this.hips[slot * 3 + axis] + this.joints[(slot * this.jointCount + joint) * 3 + axis] / 1000;
  }
}

export interface CourseState {
  broken: Set<string>;
  solid: Set<string>;
  shaking: Set<string>;
  gone: Map<string, number>;
  knockdowns: number[];
  falls: number[];
  status: string[];
  finished: (number | null)[];
}

const STATUS: Record<string, string> = { knockdown: 'down', fall: 'falling', recover: 'running', respawn: 'running' };

/** Replay the event log up to a time. Everything the course shows comes from this log. */
export function courseStateAt(meta: RaceMeta, time: number): CourseState {
  const count = meta.runners.length;
  const state: CourseState = {
    broken: new Set(), solid: new Set(), shaking: new Set(), gone: new Map(),
    knockdowns: Array(count).fill(0), falls: Array(count).fill(0),
    status: Array(count).fill('running'), finished: Array(count).fill(null),
  };
  for (const event of meta.events) {
    if (event.t > time) break;
    const tile = `${event.row}:${event.column}`;
    const door = `${event.row}:${event.door}`;
    if (event.type === 'door_break') state.broken.add(door);
    if (event.type === 'door_solid') state.solid.add(door);
    if (event.type === 'tile_shake') state.shaking.add(tile);
    if (event.type === 'tile_drop') { state.shaking.delete(tile); state.gone.set(tile, event.t); }
    if (event.runner === null) continue;
    if (event.type === 'knockdown') state.knockdowns[event.runner]++;
    if (event.type === 'fall') state.falls[event.runner]++;
    if (event.type === 'finish') state.finished[event.runner] = event.t;
    const status = STATUS[event.type];
    if (status) state.status[event.runner] = status;
  }
  return state;
}

/** Finished runners by time, then everyone else by distance. */
export function standings(recording: Recording, time: number, state: CourseState) {
  const hips: Vec3 = [0, 0, 0];
  return recording.meta.runners
    .map((runner, index) => ({ ...runner, index, x: recording.joint(time, index, 0, hips)?.[0] ?? Infinity, finished: state.finished[index] }))
    .sort((a, b) => (a.finished ?? Infinity) - (b.finished ?? Infinity) || b.x - a.x);
}

const DESCRIPTIONS: Partial<Record<string, (event: RaceEvent, name: string) => string>> = {
  door_break: (_, name) => `${name} broke through a door`,
  door_solid: (_, name) => `${name} hit a solid door`,
  arm_hit: (event, name) => `${name} was hit by the ${event.sweeper} sweeper`,
  tile_shake: (_, name) => `${name} stepped on a fake tile`,
  fall: (_, name) => `${name} fell`,
  respawn: (_, name) => `${name} is back at the checkpoint`,
  finish: (event, name) => `${name} finished, place ${event.place}`,
};

export function describe(event: RaceEvent, meta: RaceMeta) {
  const write = DESCRIPTIONS[event.type];
  return write && event.runner !== null ? write(event, meta.runners[event.runner].name) : null;
}
