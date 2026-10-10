# pick-101 contact variant

`so101_pick101_contact.xml` is adapted from
[`ggand0/pick-101`](https://github.com/ggand0/pick-101) commit
`c9d5400ffa0b742fe58a65750fed7982008450d9`. Its MIT notice is in
`LICENSE-PICK101-MIT`. The contact model adds two finger pads, fingertip
sites, a wrist camera, gripper force range, adjacent-body collision
exclusions, and a centered tool site. We disabled visual-mesh contacts and
the simulator disables the robot's full collision meshes in this variant;
the two added finger pads remain collision-active. This prevents unstable
full-mesh collisions with our tabletop scene. The local
`scene_contact_pick_place.xml` uses this model with our table, tray, and
three task objects plus pick-101's timestep, elliptic friction, and
no-slip settings.

The original `scene_pick_place.xml` remains the training and primary
evaluation scene. Use `SO101PickPlaceEnv(physics_variant="pick101_contact",
assist_grasp=False)` to run a stricter contact-based transfer check.
Its results must be reported separately because the grasp physics and tool
site differ from the training scene.
