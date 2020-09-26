import { Signal } from "./signaling";
import { DOMView } from "./dom_view";
import { Keys } from "./dom";
import { PlotView } from "../models/plots/plot";
import { Toolbar } from "../models/tools/toolbar";
import { ToolView } from "../models/tools/tool";
export declare type GestureEvent = {
    type: "pan" | "panstart" | "panend" | "pinch" | "pinchstart" | "pinchend";
    sx: number;
    sy: number;
    deltaX: number;
    deltaY: number;
    scale: number;
    shiftKey: boolean;
};
export declare type TapEvent = {
    type: "tap" | "doubletap" | "press";
    sx: number;
    sy: number;
    shiftKey: boolean;
};
export declare type MoveEvent = {
    type: "mousemove" | "mouseenter" | "mouseleave";
    sx: number;
    sy: number;
};
export declare type ScrollEvent = {
    type: "wheel";
    sx: number;
    sy: number;
    delta: number;
};
export declare type UIEvent = GestureEvent | TapEvent | MoveEvent | ScrollEvent;
export declare type KeyEvent = {
    type: "keyup" | "keydown";
    keyCode: Keys;
};
export declare type EventType = "pan" | "pinch" | "rotate" | "move" | "tap" | "press" | "scroll";
export declare type UISignal<E> = Signal<{
    id: string | null;
    e: E;
}, UIEvents>;
export declare class UIEvents implements EventListenerObject {
    readonly plot_view: PlotView;
    readonly toolbar: Toolbar;
    readonly hit_area: HTMLElement;
    readonly pan_start: UISignal<GestureEvent>;
    readonly pan: UISignal<GestureEvent>;
    readonly pan_end: UISignal<GestureEvent>;
    readonly pinch_start: UISignal<GestureEvent>;
    readonly pinch: UISignal<GestureEvent>;
    readonly pinch_end: UISignal<GestureEvent>;
    readonly rotate_start: UISignal<GestureEvent>;
    readonly rotate: UISignal<GestureEvent>;
    readonly rotate_end: UISignal<GestureEvent>;
    readonly tap: UISignal<TapEvent>;
    readonly doubletap: UISignal<TapEvent>;
    readonly press: UISignal<TapEvent>;
    readonly move_enter: UISignal<MoveEvent>;
    readonly move: UISignal<MoveEvent>;
    readonly move_exit: UISignal<MoveEvent>;
    readonly scroll: UISignal<ScrollEvent>;
    readonly keydown: UISignal<KeyEvent>;
    readonly keyup: UISignal<KeyEvent>;
    private readonly hammer;
    constructor(plot_view: PlotView, toolbar: Toolbar, hit_area: HTMLElement);
    destroy(): void;
    handleEvent(e: KeyboardEvent): void;
    protected _configure_hammerjs(): void;
    register_tool(tool_view: ToolView): void;
    private _register_tool;
    protected _hit_test_renderers(sx: number, sy: number): DOMView | null;
    protected _hit_test_frame(sx: number, sy: number): boolean;
    protected _hit_test_canvas(sx: number, sy: number): boolean;
    _trigger<E extends UIEvent>(signal: UISignal<E>, e: E, srcEvent: Event): void;
    trigger<E>(signal: UISignal<E>, e: E, id?: string | null): void;
    protected _trigger_bokeh_event(e: UIEvent): void;
    private _get_sxy;
    private _gesture_event;
    private _tap_event;
    private _move_event;
    private _scroll_event;
    private _key_event;
    private _pan_start;
    private _pan;
    private _pan_end;
    private _pinch_start;
    private _pinch;
    private _pinch_end;
    private _rotate_start;
    private _rotate;
    private _rotate_end;
    private _tap;
    private _doubletap;
    private _press;
    private _mouse_enter;
    private _mouse_move;
    private _mouse_exit;
    private _mouse_wheel;
    private _key_down;
    private _key_up;
}
