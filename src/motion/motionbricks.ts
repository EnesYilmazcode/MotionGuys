import type { MotionProvider, MotionTarget, MotionFrame, MotionChunk } from './provider.ts';

export class MotionBricksProvider implements MotionProvider {
  async generate(_target: MotionTarget, _context: MotionFrame[]): Promise<MotionChunk> {
    throw new Error('MotionBricks is not connected yet. See docs/MOTIONBRICKS.md.');
  }
}
