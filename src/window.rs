use std::cell::RefCell;
use std::rc::Rc;
//use gtk4 as gtk;
//use gtk::prelude::*;
//use gtk::{glib, Application, ApplicationWindow};
//use plotters_gtk4::Paintable;
use crate::sim::cauchy_fvm::CauchyFVM;
use crate::mesh::TriangleMesh;
//use crate::plotting;
use ndarray::array;
use std::sync::{Arc, Mutex};

pub fn create_sim_window_threaded(sim: Arc<Mutex<CauchyFVM>>) -> glib::ExitCode {
    let app = Application::builder().
        application_id("org.example.Simulator").
        build();

    app.connect_activate(move |app| {
        // create main window
        let window = ApplicationWindow::builder().
            application(app).
            default_width(640).
            default_height(480).
            title("Simulation").
            build();
        let paintable = Paintable::new((640,480)); 
        let image = gtk::Picture::for_paintable(&paintable);
        window.set_child(Some(&image));
         
        let paintable = paintable.clone();
        
        let sim_clone = sim.clone();
        // constant timed intervals
        // 16ms = 60fps
        glib::timeout_add_local(std::time::Duration::from_millis(33), move || {
            let sim = sim_clone.lock().unwrap();
            sim.display_on_paintable(&paintable); 
            glib::ControlFlow::Continue
        });

        // show window
        window.present();
    });  
    app.run()
}

pub fn create_sim_window() -> glib::ExitCode {
    let app = Application::builder().
        application_id("org.example.Simulator").
        build();

    app.connect_activate(|app| {
        // create main window
        let window = ApplicationWindow::builder().
            application(app).
            default_width(640).
            default_height(480).
            title("Simulation").
            build();
        let paintable = Paintable::new((640,480)); 
        let image = gtk::Picture::for_paintable(&paintable);
        window.set_child(Some(&image));
        
        // Create triangle mesh
        let tmesh = TriangleMesh::new_beam(6.0, 2.0, (12, 4));
        
        // Create refcell simulator
        let sim = Rc::new(RefCell::new(CauchyFVM::new(&tmesh, "rubber", 1e-3)));
        
        {
            let paintable = paintable.clone();
            let sim = sim.clone();
           
            // draw when ready
            glib::idle_add_local(move || {
                sim.borrow_mut().update();
                sim.borrow_mut().display_on_paintable(&paintable);
                if sim.borrow_mut().t > 5.0 {
                    sim.borrow_mut().
                        set_immovable_boundary("leftright");
                    sim.borrow_mut().
                        set_traction_boundary("down");
                    sim.borrow_mut().
                        set_traction_force(array![0.0, -1e6]);


                } 
                glib::ControlFlow::Continue
            });

            // constant timed intervals
            /* 33ms = 30fps
            glib::timeout_add_local(std::time::Duration::from_millis(33), move || {
                sim.borrow_mut().update();
                plotting::draw_triangle_mesh_on_area(&sim.borrow().sim_mesh, &paintable);
                glib::ControlFlow::Continue
            });
            */
        }

        // show window
        window.present();
    }); 
    
    app.run()

}
