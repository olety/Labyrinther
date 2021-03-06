import os
from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template,
    render_template_string,
)
from genetic.genetic import GeneticAlgorithm, Labyrinth
import numpy as np
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(1)
app = Flask(__name__)
if not os.path.exists("static/"):
    os.makedirs("static/")


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        with open("static/count.txt", "w+") as f:
            i = f.readline()
            # TODO: Optimize this?
            if i == "":
                i = 0
            else:
                i = int(i) + 1
            f.seek(0)
            f.write(str(i))
            f.truncate()
        executor.submit(
            GeneticAlgorithm(
                labyrinth=Labyrinth(
                    file_obj=request.files.get(request.form.get("labyrinth"))
                ),
                num_population=request.form.get("pop"),
                max_iter=request.form.get("iters"),
                crossover_rate=request.form.get("crossover_rate"),
                crossover_pts=request.form.get("crossover_pts"),
                mutation_rate=request.form.get("mutation_rate"),
                selection=request.form.get("selection"),
                elitism_num=request.form.get("elites"),
                min_moves_mult=request.form.get("min_moves_mult"),
                max_moves_mult=request.form.get("max_moves_mult"),
            ).save_data(
                file_dir=os.path.join(app.static_folder, str(i)),
                pic_last_plot=True,
                dyn_avg_fit=True,
                dyn_last_fit=True,
            )
        )
        return redirect("/{}".format(i))
    else:
        return render_template("index.html")


@app.route("/<report_id>", methods=["GET"])
def show_plots(report_id):
    arr_path = os.path.join(app.static_folder, "{}".format(report_id), "arr.npy")
    if os.path.isfile(arr_path):
        (
            max_gen,
            max_iter,
            best_moveset,
            selection,
            avg_fitness,
            setup,
            found_winner,
        ) = np.load(arr_path, allow_pickle=True)
        try:
            script1, div1 = np.load(
                os.path.join(
                    app.static_folder, "{}".format(report_id), "dyn_last_fit.npy"
                ),
                allow_pickle=True,
            )
            script2, div2 = np.load(
                os.path.join(
                    app.static_folder, "{}".format(report_id), "dyn_avg_fit.npy"
                ),
                allow_pickle=True,
            )
        except FileNotFoundError as e:
            print(e)
        # Change this to load from file if it gets too big
        # plot_table_conf = {'Last moveset (pic)': 'last.png',
        #                    'Full algorithm (gif)': 'full.gif',
        #                    'Last moveset (gif)': 'last.gif'}
        # plot_urls = dict()
        # plot_urls['names'] = [name for name, filename in plot_table_conf.items()]
        # plot_urls['links'] = [url_for('static', filename='{}/{}'.format(report_id, filename))
        #                       for name, filename in plot_table_conf.items()]
        return render_template(
            "plots.html".format(report_id),
            script=script1,
            div=div2,
            script2=script2,
            div2=div1,
            result_moveset=url_for("static", filename="{}/last.png".format(report_id)),
            id=report_id,
            setup=setup,
            num_tries=max_gen,
            num_tries_max=max_iter,
            found_winner=found_winner,
            winner_moveset=zip(
                str(best_moveset).split(" "), best_moveset.move_string_pairs
            ),
        )
    else:
        return render_template_string("Wait for the process to finish")


if __name__ == "__main__":
    app.run(threaded=True)
