height = 20;
wall_thickness = 2;
width = 50;
length = 70;
radius = width/2;

module base(){
    cube ([width, width, height], center=true);
    translate([0, radius, 0]){
        cylinder(h=height, r=radius, center=true, $fn=100);
    }
}


module ventilation () {
    union() {
        difference() {
            base();
            sx = (width - 2*wall_thickness) / width;
            sy = (length - 2*wall_thickness) / length;
            scale([sx, sy, 1])
            translate([0, 0, -1]) {
                base();
            }
            for (i=[-5:5]) {
            translate([i*wall_thickness*2, -width/2, 0])
                cube([wall_thickness, 2*wall_thickness, height-3*wall_thickness], center=true);
            }
        }
        translate([7.5, 0, 0])
            cube([width-15, wall_thickness, height], center=true);
        translate([-7.5, -15, 0])
            cube([width-15, wall_thickness, height], center=true);
    }
}
    
ventilation();
translate([0, 80, 0]){
    ventilation();
}
translate([60, 0, 0]){
    ventilation();
    translate([0, 80, 0]){
        ventilation();    
    }
}