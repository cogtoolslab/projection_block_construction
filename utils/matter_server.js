global.__base = __dirname + "/";

const argv = require("minimist")(process.argv.slice(2));
// app = require("express")(),
// express = require("express"),
// _ = require("lodash");

var port = argv.port || 5069;

var tcp_fallback = argv.tcp || false; // are we on a Windows system? If so, we need to use the TCP fallback

// var debug = argv.debug || false;
var debug = argv.debug || false;

var busy = false; // this is true while we're executing a request to prevent messing up the world

var zmq = require("zeromq");
var socket = zmq.socket("rep"); //using a reply socket here

//create the server
if (tcp_fallback) {
  if (debug) {
    console.log("Using TCP fallback. Binding sync...");
  }
  socket.bindSync("tcp://localhost:" + port);
  if (debug) {
    console.log("Socket bound to", "tcp://localhost:" + port);
  }
} else {
  if (debug) {
    console.log("Using ICP. Binding sync");
  }
  socket.bindSync("ipc:///tmp/matter_server_" + port);
  if (debug) {
    console.log("Socket bound to", "ipc:///tmp/matter_server_" + port);
  }
}

socket.on("message", function (msg) {
  while (busy) {} // wait until we're not busy
  if (debug) {
    console.log("Received message: " + msg.toString());
  }
  var busy = true;
  var data = JSON.parse(msg.toString());
  var blocks = parseBlocks(data);
  if (debug) {
    console.log(blocks);
  }
  setupWorldWithBlocks(blocks);
  var stable = checkStability(data);
  //send result as bool
  socket.send(stable);
  busy = false;
});


var parseBlocks = function (data) {
  // blocks have x, y, w, h
  // for matter, y is upper corner of area. 0,0 is top left, with y decreasing as we go up the area
  // left is x = 0
  // positions specify the midpoint of a rectangle
  var blocks = [];
  for (var i = 0; i < data.length; i++) {
    obj = data[i];
    block = {
      x: x_to_coord(Number(obj.x), Number(obj.w)),
      y: y_to_coord(Number(obj.y), Number(obj.h)),
      w: Number(obj.w),
      h: Number(obj.h),
    };
    blocks.push(block);
  }
  return blocks;
};

var setupWorldWithBlocks = function (blocks) {
  // clear current world
  World.clear(engine.world);
  // add the floor
  ground = new Block.Boundary(
    canvasWidth / 2,
    floorY,
    canvasWidth * 1.5,
    floorHeight
  );
  // add each block to the world
  for (var i = 0; i < blocks.length; i++) {
    var block = blocks[i];
    var b = Bodies.rectangle(
      block.x,
      block.y,
      block.w * sF,
      block.h * sF,
      Block.options
    );
    World.add(engine.world, b); //this where a block gets added to matter
  }
};

const MOVEMENT_DELTA = 10;
const ANGLE_DELTA = 0.7;
const SIM_TIME = 3500; //3.5 seconds is what the timeout in the browser is set to
const FRAME_LENGTH = 1000 / 60;

var checkStability = function () {
  // get the starting positions of all blocks
  var start_positions = [];
  for (var i = 0; i < world.bodies.length; i++) {
    var block = world.bodies[i];
    start_positions.push({
      x: block.position.x,
      y: block.position.y,
      angle: block.angle,
    });
  }
  // we need to run the engine in increments, long time steps break it
  for (var t = 0; t < SIM_TIME / FRAME_LENGTH; t++) {
    Engine.update(engine, FRAME_LENGTH);
    // check if all blocks are in the same place
    for (var i = 1; i < world.bodies.length; i++) {
      var block = world.bodies[i];
      var start_position = start_positions[i];
      // check if block has moved
      var xMove =
        Math.abs(block.position.x - start_position.x) > MOVEMENT_DELTA;
      var yMove =
        Math.abs(block.position.y - start_position.y) > MOVEMENT_DELTA;
      var rotated = Math.abs(block.angle - start_position.angle) > ANGLE_DELTA;
      if (xMove || yMove || rotated) {
        return false;
      }
    }
  }
  return true;
};

// ---- set up matter world ----

var Matter = require("matter-js");

var Block = require("../utils/block.js");

// Aliases for Matter functions—made global for imported functions
(global.Engine = Matter.Engine),
  (global.World = Matter.World),
  (global.Bodies = Matter.Bodies),
  (global.Constraint = Matter.Constraint),
  (global.MouseConstraint = Matter.MouseConstraint),
  (global.Mouse = Matter.Mouse),
  (global.Sleeping = Matter.Sleeping),
  (global.Runner = Matter.Runner);

// Environment parameters
var canvasHeight = 450;
var canvasWidth = 450;
var menuHeight = canvasHeight / 4.2;
var menuWidth = canvasWidth;
var floorY = canvasHeight - menuHeight;
var floorHeight = canvasHeight / 3;
var aboveGroundProp = floorY / canvasHeight;

// Stimulus parameters
var stimCanvasWidth = canvasWidth;
var stimCanvasHeight = canvasHeight;
var stimX = stimCanvasWidth / 2;
var stimY = stimCanvasHeight / 2;

// Scaling values
var sF = 25; //scaling factor to change appearance of blocks
global.worldScale = 2.2; //scaling factor within matterjs

// Global Variables for Matter js and custom Matter js wrappers
var engine;

// Set up Matter Physics Engine
engineOptions = {
  enableSleeping: true,
  velocityIterations: 24,
  positionIterations: 12,
};

world = World.create({
  gravity: {
    y: 2,
  },
});
global.engine = Engine.create(engineOptions);
var engine = global.engine;
engine.world = world;

if (debug) {
  console.log("Created engine & world");
}

var x_to_coord = function (x, w) {
  //okay, so I think the x, y coordinates mark the middle of the rectangle, so we need to take the height into account
  // 137.5 determined empirically (but it also doesn't really matter)
  return 137 + x * sF - (w * sF / 2);
};

var y_to_coord = function (y, h) {
  // first term is merely the y position of the floor
  return (
    floorY * worldScale - (floorHeight * worldScale) / 2 - y * sF - (h * sF) / 2
  );
};
