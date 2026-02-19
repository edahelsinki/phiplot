from __future__ import annotations
import argparse
import logging
import pathlib
from typing import TYPE_CHECKING, Any
import uuid
from pathlib import Path
import panel as pn
from phiplot.modules import EmbeddingHandler, DataHandler, WebUI

pn.extension("mathjax")

if TYPE_CHECKING:
    from phiplot.main import RuntimeManager

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class App:
    """
    Core backend application that wires together data access, summaries,
    clustering, embeddings, interactive plotting, and the web-based user
    interface for a single browser session.

    Each `App` instance is associated with a unique `session_id` 
    and is created and managed by `RuntimeManager`.

    Args:
        runtime_manager (RuntimeManager): Manager responsible for
            creating and tracking application instances and shared resources.
        session_id (str): Unique identifier for this application instance.
    """

    def __init__(self, runtime_manager: RuntimeManager, session_id: str):
        self.runtime_manager = runtime_manager
        self.session_id = session_id

        self.embedding_handler = EmbeddingHandler()
        self.data_handler = DataHandler(
            self.runtime_manager.local_db,
            self.embedding_handler
        )

        self.ui = WebUI(self, runtime_manager.developer, "default")
        self.ui.build()
        
        current_cookie = pn.state.cookies.get("phiplot_session_id", None)
        if self.session_id != current_cookie:
            self.set_id_cookie()

    def set_id_cookie(self) -> None:
        """
        Store the current `session_id` as a browser cookie.
        The lifetime of the cookie is one day.
        """

        cookie = pn.pane.HTML(f"""
        <script>
            const expires = new Date(Date.now() + 24 * 60 * 60 * 1000).toUTCString()
            document.cookie = "phiplot_session_id={self.session_id}; Expires=${{expires}}, SameSite=None; Secure";
        </script>
        """)
        self.ui.header.append(cookie)

    def clear_id_cookie(self) -> None:
        """
        Explicilty expire the `session_id` cookie in the browser.
        """

        cookie = pn.pane.HTML("""
        <script>
            // Expire the cookie by setting a past date
            document.cookie = "phiplot_session_id=; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/; SameSite=None; Secure";
        </script>
        """)
        self.ui.header.append(cookie)

    def restart(self) -> None:
        """
        Restart the application by destroying the current session
        starting a new one.

        Args:
            session_id (str): Identifier of the session to remove.
        """

        self.runtime_manager.destroy_session(self.session_id)
        pn.state.location.reload = True

    def view(self) -> pn.viewable.Viewable:
        """
        Return the root Panel view for this application instance.
        """

        return self.ui.view

class RuntimeManager:
    """
    Manage application runtime and session-scoped `App` instances.

    Maintains a mapping from `session_id` values to `App` instances and is 
    responsible for creating new apps when a session starts and configuring 
    the Panel server and shared resources.

    Args:
        developer (bool): If True, enable developer mode when serving the 
            application. Defaults to False.
        local_db (bool): If True, configure the application to use a local
            database backend instead of a remote one. Defaults to False.
    """

    def __init__(self, developer: bool = False, local_db: bool = False):
        self.developer = developer
        self.local_db = local_db
        self._sessions: dict[str, 'App'] = {}
        
        # Shared directory for molecular 2D structures
        self._mol_dir_path = pathlib.Path("phiplot/assets/mol_structures").resolve()
        self._mol_dir_path.mkdir(parents=True, exist_ok=True)

        # Shared directory for static media assets
        self._fig_dir_path = pathlib.Path("phiplot/assets/media").resolve()
        self._fig_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Shared separate DOCS page hosted by the server
        docs_path = Path("phiplot/assets/docs/DOCS.md").resolve()
        text = docs_path.read_text(encoding="utf-8")
        self.docs_md = pn.pane.Markdown(text, sizing_mode="stretch_both")

        self._static_dirs = {
            "phiplot/assets/mol_structures": str(self._mol_dir_path),
            "phiplot/assets/media": str(self._fig_dir_path)
        }

    def serve(self, port: int | None = None, address: str | None = None, num_procs: int = 1) -> None:
        """
        Start the Panel server to host the web application.

        Args:
            port (int): Port to listen on. Defaults to 5006 when None
            address (str | None): Address to bind to. Defaults to "localhost" when None.
            num_procs (int): Number of processes to start. Defaults to 1.
        """
        
        # Use port 5006 and localhost if None is provided
        port = port or 5006
        address = address or "localhost"

        # Explicitly specify the allowed websocket origins to access the application.
        # If binding to address "0.0.0.0" (all interfaces), allow all origins with ["*"].
        # Otherwise, restrict to the specific "address:port" origin for security.
        if address == "0.0.0.0":
            origins = ["*"]
        else:
            origins = [f"{address}:{port}"]

        pn.serve(
            {"": self._get_view, "docs": self.docs_md},
            port=port,
            address=address,
            show=False,
            static_dirs=self._static_dirs,
            autoreload=self.developer,
            use_reloader=self.developer,
            allow_websocket_origin=origins,
            num_procs=num_procs,
            sessions_per_process=1,
            unused_session_lifetime=30 * 60 * 1000
        )

    def destroy_session(self, session_id) -> None:
        """
        Remove the application session and its associated resources.

        Args:
            session_id (str): Identifier of the session to remove.
        """

        app = self._sessions.pop(session_id, None)
        app.clear_id_cookie()

    def _get_view(self) -> pn.template.Template:
        """
        Retrieve the view for the current browser session.

        Returns:
            pn.template.Template: The templated view for the 
                a new application with the resolved state.
        """

        cookies = pn.state.cookies
        session_id = cookies.get("phiplot_session_id", str(uuid.uuid4()))
        app = App(self, session_id)
        self._sessions[session_id] = app
            
        return app.view()

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the runtime.

    Returns:
    argparse.Namespace: Parsed arguments, including:
        - developer (bool): Whether to run in developer mode.
        - localdb (bool): Whether to use a local database backend.
        - port (int | None): Port to listen on.
        - address (str | None): Address to bind to.
        - num_procs (int | None): Number of processes to start.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--developer", action="store_true", help="Use developer mode.")
    parser.add_argument("--localdb", action="store_true", help="Use a local database instance.")
    parser.add_argument("--port", type=int, help="Port to listen on.")
    parser.add_argument("--address", type=str, help="Address to listen on.")
    parser.add_argument("--num_procs", type=int, help="Number of processes to start.", default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    runtime_manager = RuntimeManager(args.developer, args.localdb)
    runtime_manager.serve(args.port, args.address, args.num_procs)