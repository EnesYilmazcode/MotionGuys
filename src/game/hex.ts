import type { RaceEvent } from './playback.ts';
import type { Shot, SpleefMeta } from './spleef.ts';

export interface HexArena {
  layers: number;
  layerHeight: number;
  hexRadius: number;
  slimeDepth: number;
  touchSeconds: number;
  countdownSeconds: number;
  pedestalHeight: number;
  radii: number[];
  tiles: [number, number][][];
  pedestals: { x: number; y: number }[];
}
export type HexMeta = Omit<SpleefMeta, 'arena'> & { arena: HexArena };

export const hexKey = (layer: number, q: number, r: number) => `${layer}:${q}:${r}`;

export function hexCenter(q: number, r: number, radius: number): [number, number] {
  return [Math.sqrt(3) * radius * (q + r / 2), 1.5 * radius * r];
}

export interface TileTimes { touched: number; gone: number; }

/** When each tile was touched and when it vanished. */
export function tileTimes(meta: HexMeta) {
  const times = new Map<string, TileTimes>();
  for (const event of meta.events) {
    if (event.type !== 'tile_touch' && event.type !== 'tile_gone') continue;
    const key = hexKey(event.layer as number, event.q as number, event.r as number);
    const entry = times.get(key) ?? { touched: Infinity, gone: Infinity };
    if (event.type === 'tile_touch') entry.touched = event.t;
    else entry.gone = event.t;
    times.set(key, entry);
  }
  return times;
}

/** The footage holds a tile's color for about 47% of its timer, then ramps to white and cuts it. */
export function tileWhiteness(time: number, entry: TileTimes | undefined, touchSeconds: number) {
  if (!entry || time < entry.touched) return 0;
  if (time >= entry.gone) return -1;
  return Math.min(Math.max((time - entry.touched) / touchSeconds - 0.47, 0) / 0.4, 1);
}

// In the file order of the roster. Short enough to read while the round moves.
const OUT_LINES = [
  'Optimus is in the slime. Deliveries now expected next year.',
  'Figure is in the slime. The demo video will be edited.',
  'G1 is in the slime. Still the cheapest one here.',
  'H1 is in the slime. Top speed did not help.',
  'Apollo is in the slime. Houston has a problem.',
  'Atlas is in the slime. Retiring again.',
  'T1 is in the slime. Lightest one here, floats well.',
  'Petman is in the slime. Should have stayed in the lab.',
];
const WIN_LINES = [
  'Optimus wins. A stock split is rumored.',
  'Figure wins. Valuation doubled.',
  'G1 wins. Ships in 45 days.',
  'H1 wins. Sprinted the whole time.',
  'Apollo wins. One small step.',
  'Atlas wins. Backflip pending.',
  'T1 wins. Smallest robot on the floor.',
  'Petman wins. The original.',
];
const FALL_LINES = [
  (name: string) => `${name} fell. Blaming the teleoperator.`,
  (name: string) => `${name} found a shortcut downward.`,
  (name: string) => `${name} missed the floor.`,
  (name: string) => `${name} is testing gravity.`,
  (name: string) => `${name} is on the next floor now.`,
  (name: string) => `${name} needs a firmware update.`,
];

const KNOCK_LINES = [
  (name: string, by: string) => `${by} flattened ${name}. No refunds.`,
  (name: string, by: string) => `${by} tackled ${name}. HR has been notified.`,
  (name: string, by: string) => `${name} just met ${by}'s elbow.`,
  (name: string, by: string) => `${by} ran over ${name}. Autonomy is hard.`,
];
/**
 * Cut to the action. Each knockdown or hop gets a shot starting just before it, spaced so the camera
 * is not jumping constantly, and the normal spectator shots fill the time in between.
 */
export function actionShots(meta: HexMeta, shots: Shot[], lead = 0.7, hold = 3.5): Shot[] {
  const win = meta.events.find(event => event.type === 'winner')?.t ?? Infinity;
  const action: Shot[] = [];
  for (const event of meta.events) {
    if ((event.type !== 'knock' && event.type !== 'leap') || event.t >= win - 1) continue;
    const start = Math.max(event.t - lead, 0);
    if (!action.length || start - action[action.length - 1].start >= hold) action.push({ start, runner: event.runner as number });
  }
  const free = shots.filter(shot => !action.some(cut => shot.start >= cut.start && shot.start < cut.start + hold));
  return [...free, ...action].sort((a, b) => a.start - b.start);
}

export const GAIT_NAMES = ['SLOW', 'WALK', 'JOG', 'SPRINT'];

/** How many tiles each robot has burned by a time, and the gait it is in. Replayed from the event log. */
export function brainStateAt(meta: HexMeta, time: number) {
  const tiles = meta.runners.map(() => 0);
  const gait: (number | null)[] = meta.runners.map(() => null);
  for (const event of meta.events) {
    if (event.t > time) break;
    if (event.runner === null) continue;
    if (event.type === 'tile_touch') tiles[event.runner]++;
    if (event.type === 'gait') gait[event.runner] = event.gait as number;
  }
  return { tiles, gait };
}

export function describeHex(event: RaceEvent, names: string[]) {
  if (event.runner === null) return null;
  const name = names[event.runner];
  const line = Math.round(event.t * 100);
  if (event.type === 'knock') return KNOCK_LINES[line % KNOCK_LINES.length](name, names[event.by as number]);
  if (event.type === 'winner') return WIN_LINES[event.runner];
  if (event.type === 'splash') return OUT_LINES[event.runner];
  if (event.type === 'fall') return FALL_LINES[Math.round(event.t * 100) % FALL_LINES.length](name);
  return null;
}
