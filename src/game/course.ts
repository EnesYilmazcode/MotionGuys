export type Action = 'run' | 'vault' | 'duck' | 'jump';
export interface Obstacle { id: string; action: Exclude<Action, 'run'>; start: number; end: number; }

export const course = {
  width: 14,
  length: 76,
  finish: 71,
  obstacles: [
    { id: 'low-wall', action: 'vault', start: 17, end: 21 },
    { id: 'low-ceiling', action: 'duck', start: 33, end: 40 },
    { id: 'gap', action: 'jump', start: 52, end: 57 },
  ] satisfies Obstacle[],
};

export function actionAt(z: number): Action {
  return course.obstacles.find(o => z >= o.start && z <= o.end)?.action ?? 'run';
}
