# Credits

MotionGuys is an independent project. It is not affiliated with or endorsed by any company named here. Product names and robot designs belong to their owners and are used for parody and comparison. The licenses below cover each artist's model files, not those trademarks or designs.

## Robots in `public/robots/`

Every model below was changed: cut into parts per bone, moved into bone space, and scaled by `scripts/build_robots.py`. Licenses are as listed by the source.

| Robot | Model | Author | License |
|---|---|---|---|
| Optimus | [Tesla Optimus](https://sketchfab.com/3d-models/d68250b144f049eb9c8b38b54203ff9d) | mriexinger | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Figure | [Tesla optimus](https://sketchfab.com/3d-models/2fab5d31927f43729a99a6e8eaf1c7f5), repainted black and simplified. Not a real Figure model. | jjeendral36 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Atlas | [Atlas robot](https://sketchfab.com/3d-models/1b582669d125496882d96a8ac3e373ea), repainted gray | bruhepic925 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| G1 | [`unitree_g1`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) in MuJoCo Menagerie, meshes simplified | Unitree Robotics | BSD-3-Clause, below |
| H1 | [`unitree_h1`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_h1) in MuJoCo Menagerie, meshes simplified | Unitree Robotics | BSD-3-Clause, below |
| Apollo | [`apptronik_apollo`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/apptronik_apollo) in MuJoCo Menagerie, meshes simplified | Apptronik | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Petman | [PETMAN Boston Dynamics](https://sketchfab.com/3d-models/f7b3ae820d164d9fa3a114c86ded4ba4) | Okopchi | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| T1 | [`booster_t1`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/booster_t1) in MuJoCo Menagerie, meshes simplified | Booster Robotics | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Crown | [A royal crown, a prop](https://sketchfab.com/3d-models/fcd365eacab747d58b8a54ccd178bcb4), simplified | Wirtualne Muzea Malopolski | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |

## Software and data

- [MotionBricks](https://github.com/NVlabs/motionbricks) by NVIDIA generates the motion, under the NVIDIA Open Model Agreement.
- [MuJoCo](https://mujoco.org/) simulates every fall, under the Apache-2.0 license. [PyTorch](https://pytorch.org/) trains and runs the network.
- [Modal](https://modal.com/) runs it. [Three.js](https://threejs.org/) draws the replay.
- [Titan One](https://fonts.google.com/specimen/Titan+One) from Google Fonts, under the SIL Open Font License.

The hexagon round is inspired by Hex-A-Gone in Fall Guys. The arena is rebuilt from scratch from what gameplay footage shows, and no assets, logos, or characters from that game are used. See [docs/HEXAGONS.md](docs/HEXAGONS.md).

## Unitree BSD-3-Clause notice

```
Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
