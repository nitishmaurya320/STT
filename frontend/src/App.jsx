import WebcamFeed from "./components/WebcamFeed";
function App() {
  return (
    <div className="h-[100vh] overflow-hidden bg-gradient-to-br from-slate-900 via-blue-900 to-slate-950 flex items-center justify-center p-4">
      {" "}
      <WebcamFeed />{" "}
    </div>
  );
}
export default App;
