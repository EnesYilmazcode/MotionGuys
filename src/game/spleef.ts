import type { RaceEvent, RaceMeta } from './playback.ts';

export interface SpleefRunner { name: string; color: string; gear: string; bulk: number; eliminatedAt: number | null; }
export interface SpleefMeta extends Omit<RaceMeta, 'course' | 'runners'> {
  arena: { layers: number; layerHeight: number; tile: number; tiles: number; half: number; dropSeconds: number };
  runners: SpleefRunner[];
}

export interface RoundState {
  shaking: Set<string>;
  gone: Map<string, number>;
  falls: number[];
  layer: number[];
  out: (number | null)[];
  winner: number | null;
}

export const tileKey = (layer: number, column: number, row: number) => `${layer}:${column}:${row}`;

/** Replay the event log up to a time. Everything the floors show comes from this log. */
export function roundStateAt(meta: SpleefMeta, time: number): RoundState {
  const count = meta.runners.length;
  const state: RoundState = {
    shaking: new Set(), gone: new Map(), falls: Array(count).fill(0), layer: Array(count).fill(0),
    out: Array(count).fill(null), winner: null,
  };
  for (const event of meta.events) {
    if (event.t > time) break;
    const key = tileKey(event.layer as number, event.column as number, event.row as number);
    if (event.type === 'tile_shake') state.shaking.add(key);
    if (event.type === 'tile_drop') { state.shaking.delete(key); state.gone.set(key, event.t); }
    if (event.runner === null) continue;
    if (event.type === 'fall') state.falls[event.runner]++;
    if (event.type === 'land') state.layer[event.runner] = event.layer as number;
    if (event.type === 'eliminated') state.out[event.runner] = event.t;
    if (event.type === 'winner') state.winner = event.runner;
  }
  return state;
}

/** The winner, then everyone still in, then whoever lasted longest. */
export function ranking(meta: SpleefMeta, state: RoundState) {
  return meta.runners
    .map((runner, index) => ({ ...runner, index, out: state.out[index], layer: state.layer[index] }))
    .sort((a, b) => Number(b.index === state.winner) - Number(a.index === state.winner)
      || (b.out ?? Infinity) - (a.out ?? Infinity) || a.layer - b.layer);
}

const DESCRIPTIONS: Partial<Record<string, (event: RaceEvent, name: string) => string>> = {
  fall: (event, name) => event.cause === 'edge' ? `${name} ran off the edge` : `${name} fell through`,
  land: (event, name) => `${name} landed on floor ${(event.layer as number) + 1}`,
  eliminated: (_, name) => `${name} is out`,
  winner: (_, name) => `${name} wins`,
};

export function describeRound(event: RaceEvent, meta: SpleefMeta) {
  const write = DESCRIPTIONS[event.type];
  return write && event.runner !== null ? write(event, meta.runners[event.runner].name) : null;
}
