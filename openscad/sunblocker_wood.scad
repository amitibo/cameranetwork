include <MCAD/units.scad>

//
// The following values where calculated by the sunshader script (from the clouds project).
// for date: 5/8/14
//sun_angle = -36.44;
//sun_offset =  0.36696701;
// for date: 12/8/14
//sun_angle = -35.3039;
//sun_offset =  0.316264308692;
// for date: 29/9/14
//sun_angle = -32.7591492134;
//sun_offset = 0.0530506267263;
// for date: 15/4/15
//sun_angle = -34.2392365306;
//sun_offset = 0.209588226863;
// for date: 24/5/15
//sun_angle = -39.5764032041;
//sun_offset = 0.473792640857;
// for date: 4/6/15
//sun_angle = -41.0892252744;
//sun_offset = 0.52118130747;
// for Leipzig, 18/10/15
sun_angle = -53.8315722229;
sun_offset = -0.258274603021;

//
// Model sizes
// -----------
// Radius of the sun shade
sb_radius = 140;
// Width of the sun shade (should match the lens size++)
// sb_width = 30; // Used for the leopard lens
camera_radius = 12.5; // OpteamX camera
sb_width = 35; // Used for the lens bought from OpteamX
// Thicknes of the sun shade arc (5 seems ok).
sb_thickness = 5;
// Resolution. Make it small while debugging. Larger (100) when creating
// final model.
sphere_res = 50;
// Inclination and offset of the sunshader
sb_angle = sun_angle;
sb_offset = sun_offset * sb_radius;

//
// Base length
//
base_height = 5;
base_length = sb_radius*2+20;
base_width = 90;
base_max_extent = 80;

camera_hole_size = 50;

//
// Create the sun shader
//
translate([sb_offset, 0, 0])
difference() {
    //
    // The main form without holes
    //
    sphere(r=sb_radius, $fn=sphere_res);
    
    //
    // Main hole
    //
    sphere(r=sb_radius-sb_thickness, $fn=sphere_res);
    
    //
    // Remove bottom half of sphere
    //
    translate([0, 0, -sb_radius/2])
    cube([sb_radius*2, sb_radius*2, sb_radius], center=true);
    
    //
    // cut at an angle of the sun shader
    //
    rotate([0, sb_angle, 0])
    {
        translate([-sb_radius-sb_width/2, 0, 0])
        cube([sb_radius*2, sb_radius*2, sb_radius*3], center=true);
        translate([sb_radius+sb_width/2, 0, 0])
        cube([sb_radius*2, sb_radius*2, sb_radius*3], center=true);
    }
}

//
// Base1:
//
translate([0, 0, -base_height/2])
union() {
    difference() {
        translate([-base_width/2, 0, 0])
        cube([base_width, base_length, base_height], center=true);
        
        //
        // Limit the extent of the base
        //
        translate([-base_width/2 - base_max_extent, 0, 0])
        cube([base_width, base_length, base_height], center=true);
        
        cylinder(h=base_height, r=camera_radius, center=true);
    }
};

//
// Base2:
//
translate([0, 0, -2*base_height])
difference() {
    difference() {
        translate([-base_width/2 - camera_hole_size/2, 0, 0])
        cube([base_width, base_length, base_height], center=true);
        
        //
        // Limit the extent of the base
        //
        translate([-base_width/2 - base_max_extent, 0, 0])
        cube([base_width, base_length, base_height], center=true);
    }
    
    //
    // Hole of camera
    //
    cube(size=[camera_hole_size, camera_hole_size, base_height], center=true);
};