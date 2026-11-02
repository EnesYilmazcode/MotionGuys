"""Train the robots' brain by self-play, with one shared network and a personality input per robot.

Run:  modal run scripts/modal_brain.py --iterations 400 --tags base,seed2
Each tag trains in its own container and writes runs/brain/policy_TAG.pt and log_TAG.jsonl. Scored against the old hand-written
policy every few iterations. A tag from VARIANTS changes the rewards.
"""

import json
import time
from pathlib import Path

import modal
import torch
from modal_progress import bar, watch
from torch.distributions import Categorical

from hex_brain import AGENTS, GAITS, MOVE_RUN, OUT, RUN, Policy, Sim, scripted_action

GAMMA = 0.995
LAMBDA = 0.95
HORIZON = 96
ENVS = 1024
EPOCHS = 3
MINIBATCH = 32768
CLIP = 0.2
LR = 3e-4
# Things worth comparing, by tag. Anything not listed uses the defaults in hex_brain.py.
VARIANTS = {
    "base": {},
    "seed2": {},
    "bigsave": {"save_bonus": 1.0},
    "freeleap": {"plain_leap_cost": 0.05},
    "thrift": {"tile_cost": 0.08},
    "rare": {"strict_save": True, "save_bonus": 1.0, "plain_leap_cost": 0.2, "leap_cooldown": 2.0},
    "rare2": {"strict_save": True, "save_bonus": 1.0, "plain_leap_cost": 0.2, "leap_cooldown": 2.0},
    "rarer": {"strict_save": True, "save_bonus": 1.0, "plain_leap_cost": 0.4, "leap_cooldown": 2.0},
    "carry": {"strict_save": True, "save_bonus": 1.0, "plain_leap_cost": 0.2, "leap_cooldown": 2.0},
    "carry2": {"strict_save": True, "save_bonus": 1.0, "plain_leap_cost": 0.2, "leap_cooldown": 2.0},
    "calm": {"strict_save": True, "save_bonus": 0.7, "plain_leap_cost": 0.4, "leap_cooldown": 3.0},
    "saves": {"strict_save": True, "save_bonus": 1.0, "leap_cooldown": 2.0, "land_fall_cost": 0.7},
    "saves2": {"strict_save": True, "save_bonus": 1.0, "leap_cooldown": 2.0, "land_fall_cost": 0.7},
    "careful": {"strict_save": True, "save_bonus": 1.0, "leap_cooldown": 2.0, "land_fall_cost": 1.5},
    "gated": {"strict_save": True, "save_bonus": 1.0, "leap_cooldown": 2.0, "land_fall_cost": 0.7, "leap_after": 10.0, "block_sectors": 2},
    "gated2": {"strict_save": True, "save_bonus": 1.0, "leap_cooldown": 2.0, "land_fall_cost": 0.7, "leap_after": 10.0, "block_sectors": 2},
    "solo7": {"solo": True, "radii": (2.2, -1, -1, -1, -1, -1), "plain_leap_cost": 0.0, "save_bonus": 0.0, "tile_cost": 0.0, "leap_distance": (3.0, 0.5, 2.2, 3.8), "land_speed": 4.0},
    "solo19": {"solo": True, "radii": (4.2, -1, -1, -1, -1, -1), "plain_leap_cost": 0.0, "save_bonus": 0.0, "tile_cost": 0.0, "leap_distance": (3.0, 0.5, 2.2, 3.8), "land_speed": 4.0},
    "solo31": {"solo": True, "radii": (6.2, -1, -1, -1, -1, -1), "plain_leap_cost": 0.0, "save_bonus": 0.0, "tile_cost": 0.0, "leap_distance": (3.0, 0.5, 2.2, 3.8), "land_speed": 4.0},
    "solo55": {"solo": True, "radii": (8.3, -1, -1, -1, -1, -1), "plain_leap_cost": 0.0, "save_bonus": 0.0, "tile_cost": 0.0, "leap_distance": (3.0, 0.5, 2.2, 3.8), "land_speed": 4.0},
    "solo19slow": {"solo": True, "radii": (4.2, -1, -1, -1, -1, -1), "touch_seconds": 1.4, "plain_leap_cost": 0.0, "save_bonus": 0.0, "tile_cost": 0.0, "leap_distance": (3.0, 0.5, 2.2, 3.8), "land_speed": 4.0},
    "solo55slow": {"solo": True, "radii": (8.3, -1, -1, -1, -1, -1), "touch_seconds": 1.4, "plain_leap_cost": 0.0, "save_bonus": 0.0, "tile_cost": 0.0, "leap_distance": (3.0, 0.5, 2.2, 3.8), "land_speed": 4.0},
    "hop": {"strict_save": True, "save_bonus": 1.0, "leap_cooldown": 1.0, "land_fall_cost": 0.7, "leap_after": 10.0, "block_sectors": 1, "land_speed": 4.0},
    "hop2": {"strict_save": True, "save_bonus": 1.0, "leap_cooldown": 1.0, "land_fall_cost": 0.7, "leap_after": 10.0, "block_sectors": 1, "land_speed": 4.0},
    "hopcold": {"strict_save": True, "save_bonus": 1.0, "leap_cooldown": 1.0, "land_fall_cost": 0.7, "leap_after": 10.0, "block_sectors": 1, "land_speed": 4.0},
    "gated1": {"strict_save": True, "save_bonus": 1.0, "leap_cooldown": 2.0, "land_fall_cost": 0.7, "leap_after": 10.0, "block_sectors": 1},
}

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy", "modal-progress").add_local_python_source("hex_brain")
app = modal.App("motionguys-brain", image=image)
volume = modal.Volume.from_name("motionguys-cache", create_if_missing=True)
CACHE = "/cache"


WARM = {"hop": "solo55slow", "hop2": "solo55slow"}


@torch.no_grad()
def score_against_script(policy, device, rounds=256, seed=7):
    """Robot 0 runs the network, the other seven run the hand-written policy. Fair odds are 1 in 8."""
    sim = Sim(rounds, device, seed=seed)
    sight = 3.0 + 5.0 * sim.rand(rounds, AGENTS)
    noise = 0.1 + 0.8 * sim.rand(rounds, AGENTS)
    finished = torch.zeros(rounds, dtype=torch.bool, device=device)
    wins, length, placements = 0, [], []
    held = torch.zeros((rounds, AGENTS), dtype=torch.long, device=device)
    for tick in range(int(120 / 0.3) + 2):
        heading_logits, gait_logits, move_logits, _ = policy(sim.observe())
        heading = Categorical(logits=heading_logits).sample()
        gait = Categorical(logits=gait_logits).sample()
        move = Categorical(logits=move_logits).sample()
        if tick % 2 == 0:
            held = scripted_action(sim, sight, noise)
        script = held
        script_move = torch.full_like(move, MOVE_RUN)
        heading_all = script.clone()
        move_all = script_move.clone()
        heading_all[:, 0], move_all[:, 0] = heading[:, 0], move[:, 0]
        # The hand-written policy sprints.
        gait_all = torch.full_like(gait, GAITS - 1)
        gait_all[:, 0] = gait[:, 0]
        if tick % 2 == 1:
            heading_all[:, 1:] = 0
        times = sim.time.clone()
        reward, done = sim.tick(heading_all, gait_all, move_all)
        for env in torch.nonzero(done & ~finished).flatten().tolist():
            finished[env] = True
            # Only a winner collects the +5.
            wins += int(reward[env, 0].item() > 2.0)
            length.append(times[env].item())
        if finished.all():
            break
    return {"winRate": wins / rounds, "seconds": sum(length) / max(len(length), 1)}


@app.function(gpu="A10", volumes={CACHE: volume}, timeout=14000)
def train(iterations: int, seed: int, tag: str) -> tuple[str, bytes, list[dict]]:
    import io

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = Sim(ENVS, device, seed=seed, **VARIANTS.get(tag, {}))
    policy = Policy().to(device)
    # Warm start: tags in WARM begin from the network that learned to survive alone on a small floor.
    if tag in WARM:
        policy.load_state_dict(torch.load(f"{CACHE}/brain/policy_{WARM[tag]}.pt", map_location=device))
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR, eps=1e-5)
    log = []
    started = time.perf_counter()
    samples_per_iteration = HORIZON * ENVS * AGENTS

    def save():
        buffer = io.BytesIO()
        torch.save(policy.state_dict(), buffer)
        Path(f"{CACHE}/brain").mkdir(exist_ok=True)
        Path(f"{CACHE}/brain/policy_{tag}.pt").write_bytes(buffer.getvalue())
        volume.commit()
        return buffer.getvalue()

    for iteration in bar(range(iterations), desc=tag):
        progress = iteration / iterations
        entropy_weight = 0.02 * (1 - progress) + 0.002
        for group in optimizer.param_groups:
            group["lr"] = LR * (1 - 0.8 * progress)
        store = {key: [] for key in ("obs", "heading", "gait", "move", "log_prob", "value", "reward", "done", "acting", "alive")}
        with torch.no_grad():
            for _ in range(HORIZON):
                obs = sim.observe()
                heading_logits, gait_logits, move_logits, value = policy(obs)
                heading_dist, gait_dist, move_dist = Categorical(logits=heading_logits), Categorical(logits=gait_logits), Categorical(logits=move_logits)
                heading, gait, move = heading_dist.sample(), gait_dist.sample(), move_dist.sample()
                store["obs"].append(obs)
                store["heading"].append(heading)
                store["gait"].append(gait)
                store["move"].append(move)
                store["log_prob"].append(heading_dist.log_prob(heading) + gait_dist.log_prob(gait) + move_dist.log_prob(move))
                store["value"].append(value)
                store["acting"].append(sim.mode == RUN)
                alive = sim.mode != OUT
                store["alive"].append(alive)
                reward, done = sim.tick(heading, gait, move)
                store["reward"].append(reward)
                # A robot that went out this tick, or any robot when the round reset, stops here.
                stopped = done[:, None].expand_as(alive) | (sim.mode == OUT)
                store["done"].append(stopped)
            _, _, _, last_value = policy(sim.observe())
        advantages, running = [], torch.zeros_like(last_value)
        next_value = last_value
        for t in reversed(range(HORIZON)):
            keep = (~store["done"][t]).float()
            delta = store["reward"][t] + GAMMA * next_value * keep - store["value"][t]
            running = delta + GAMMA * LAMBDA * keep * running
            advantages.append(running)
            next_value = store["value"][t]
        advantages = torch.stack(advantages[::-1])
        returns = advantages + torch.stack(store["value"])
        acting = torch.stack(store["acting"])
        alive = torch.stack(store["alive"])
        pick = acting.flatten()
        obs = torch.stack(store["obs"]).flatten(0, 2)
        heading, gait, move = torch.stack(store["heading"]).flatten(), torch.stack(store["gait"]).flatten(), torch.stack(store["move"]).flatten()
        old_log_prob = torch.stack(store["log_prob"]).flatten()
        flat_returns, flat_advantages = returns.flatten(), advantages.flatten()
        adv_mean, adv_std = flat_advantages[pick].mean(), flat_advantages[pick].std() + 1e-8
        value_mask = alive.flatten()
        count = obs.shape[0]
        for _ in range(EPOCHS):
            order = torch.randperm(count, device=device)
            for start in range(0, count, MINIBATCH):
                index = order[start:start + MINIBATCH]
                heading_logits, gait_logits, move_logits, value = policy(obs[index])
                heading_dist, gait_dist, move_dist = Categorical(logits=heading_logits), Categorical(logits=gait_logits), Categorical(logits=move_logits)
                log_prob = heading_dist.log_prob(heading[index]) + gait_dist.log_prob(gait[index]) + move_dist.log_prob(move[index])
                ratio = (log_prob - old_log_prob[index]).exp()
                advantage = (flat_advantages[index] - adv_mean) / adv_std
                mask = pick[index].float()
                surrogate = torch.min(ratio * advantage, ratio.clamp(1 - CLIP, 1 + CLIP) * advantage)
                policy_loss = -(surrogate * mask).sum() / mask.sum().clamp(min=1)
                entropy = ((heading_dist.entropy() + gait_dist.entropy() + move_dist.entropy()) * mask).sum() / mask.sum().clamp(min=1)
                v_mask = value_mask[index].float()
                value_loss = (((value - flat_returns[index]) ** 2) * v_mask).sum() / v_mask.sum().clamp(min=1)
                loss = policy_loss + 0.5 * value_loss - entropy_weight * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
        if iteration % 20 == 0 or iteration == iterations - 1:
            rounds = max(sim.stats["rounds"], 1.0)
            stats = {key: value / rounds for key, value in sim.stats.items() if key not in ("rounds", "round_seconds")}
            round_seconds = sim.stats["round_seconds"] / rounds
            sim.stats = {key: 0.0 for key in sim.stats}
            scored = score_against_script(policy, device)
            entry = {
                "iteration": iteration, "minutes": round((time.perf_counter() - started) / 60, 2),
                "samples": (iteration + 1) * samples_per_iteration, "entropy": round(entropy.item(), 3),
                "reward": round(torch.stack(store["reward"]).mean().item(), 4), "tag": tag, "roundSeconds": round(round_seconds, 1), "perRound": {k: round(v, 1) for k, v in stats.items()},
                **{k: round(v, 3) for k, v in scored.items()},
            }
            log.append(entry)
            print(json.dumps(entry), flush=True)
            save()
    return tag, save(), log


@app.local_entrypoint()
def main(iterations: int = 400, tags: str = "base"):
    out = Path(__file__).resolve().parent.parent / "runs" / "brain"
    out.mkdir(parents=True, exist_ok=True)
    names = tags.split(",")
    calls = [(iterations, index + 1, name) for index, name in enumerate(names)]
    failures = []
    with watch(app, name="train") as progress:
        for result in progress.track(train.starmap(calls, return_exceptions=True), total=len(calls)):
            if isinstance(result, BaseException):
                failures.append(result)
                print("FAILED", result)
                continue
            tag, weights, log = result
            (out / f"policy_{tag}.pt").write_bytes(weights)
            (out / f"log_{tag}.jsonl").write_text("\n".join(json.dumps(entry) for entry in log))
            print("final", log[-1])
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(names)} runs failed")
