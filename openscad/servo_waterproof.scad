wall_thick = 4;

tolerance = 0.5;

//
// Servo definition.
//
servo_length = 41.8 + 2*tolerance;
servo_width = 20.2 + tolerance;
servo_half_height = 30.5 + tolerance;
servo_height = 42.9 + tolerance;

extra = 12;
extra_h = 14;

//
// Hole definitions.
//
hole_radius = 1.6;
hole_dist1 = 9.8;
hole_dist2 = 51.1;

//
// Outer definitions.
//
body_length = servo_length + 2*extra;
body_width = 39;

//
// Hole spaces.
//
screw_width = 6.5;
screw_height = 3;
accuracy = 100;

//
// Space.
//
space_dist1 = 5+screw_height;


//
// cover
//
cover_height = 1;

module hole(){
     cylinder(h=servo_height, r=hole_radius, center=false, $fn=accuracy);
};


module holes(){
     translate([-hole_dist2/2, -hole_dist1/2, 0]){
          union(){
               hole();
               translate([51.1, 0, 0]){
                    hole();
               };
               translate([0, hole_dist1, 0]){
                    hole();
                    translate([51.1, 0, 0]){
                         hole();
                    };
               };
          };
     };
};


module space(){
     translate([0, -screw_width/2, 0])
          cube([body_length, screw_width, screw_height]);
};


module spaces(){
     translate([0, -hole_dist1/2, 0]){
          union(){
               space();
               translate([0, hole_dist1, 0]){
                    space();
               };
          };
     };
};


difference(){
     //
     // Body.
     //
     union(){
          //
          // Back of servo
          //
          cube([body_length, body_width, wall_thick]);

          //
          // Top of servo
          //
          cube([body_length, wall_thick, wall_thick+servo_height]);

          //
          // Rain cover
          //
          translate([0, 0, wall_thick+servo_height])
               cube([body_length, wall_thick+cover_height, cover_height]);

          //
          // Two supports
          //
          translate([body_length-extra, 0, 0])
               cube([extra, wall_thick+servo_width, wall_thick+servo_half_height]);
          translate([0, 0, wall_thick+servo_half_height-extra_h])
               cube([extra, wall_thick+servo_width, extra_h]);
     };

     //
     // holes.
     //
     translate([body_length/2, wall_thick+servo_width/2, 0])
          holes();

     //
     // Space.
     //
     translate([0, wall_thick+servo_width/2, wall_thick+servo_half_height-space_dist1])
          spaces();
};
