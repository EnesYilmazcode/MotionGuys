import type { Action } from '../game/course.ts';

// Project-owned contract, not the upstream MotionBricks API.
export interface MotionTarget {
  runnerId: number;
  action: Action;
  position: [number, number, number]; // meters, Y up
  facing: [number, number, number];
  speed: number;
}
export interface JointPose {
  name: string;
  position: [number, number, number];
  rotation: [number, number, number, number]; // quaternion XYZW
}
export interface MotionFrame { time: number; joints: JointPose[]; }
export interface MotionChunk { runnerId: number; frames: MotionFrame[]; }
export interface MotionProvider {
  generate(target: MotionTarget, context: MotionFrame[]): Promise<MotionChunk>;
}
